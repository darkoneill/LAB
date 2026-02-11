"""
Swarm Mode - Specialized Sub-Agent System.

Instead of a single generalist agent, the Orchestrator can spawn
specialized agent instances with tailored system prompts and permissions:

- Coder Agent: Expert Python coder with full sandbox R/W access.
- Reviewer Agent: Security reviewer with read-only sandbox access.
- Researcher Agent: Focused on information gathering and analysis.

Flow: Coder writes code -> Reviewer critiques -> Coder corrects -> Orchestrator validates.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from openclaw.config.settings import get_settings
from openclaw.tools.repo_map import generate_repo_map

if TYPE_CHECKING:
    from openclaw.agent.brain import AgentBrain

logger = logging.getLogger("openclaw.agent.swarm")


class AgentRole(str, Enum):
    """Predefined specialized agent roles."""
    CODER = "coder"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    TESTER = "tester"
    SECURITY = "security"


# ── Agent Profile Definitions ────────────────────────────────────────

AGENT_PROFILES: dict[str, dict] = {
    AgentRole.CODER: {
        "name": "Coder Agent",
        "system_prompt": (
            "Tu es un expert Python strict et rigoureux. "
            "Tu ecris du code propre, efficace et bien documente. "
            "Tu as acces complet au sandbox en lecture et ecriture. "
            "Reponds UNIQUEMENT avec du code executable. "
            "Pas d'explications sauf en commentaires dans le code."
        ),
        "temperature": 0.3,
        "max_tokens": 4096,
        "sandbox_access": "read_write",
        "tools": ["python", "shell", "write_file", "patch_file"],
    },
    AgentRole.REVIEWER: {
        "name": "Reviewer Agent",
        "system_prompt": (
            "Tu es un expert en securite et qualite de code. "
            "Tu analyses le code fourni pour detecter : "
            "1. Failles de securite (injection, XSS, SSRF, etc.) "
            "2. Bugs logiques et cas limites non geres "
            "3. Problemes de performance "
            "4. Non-respect des bonnes pratiques "
            "Tu as acces en LECTURE SEULE au sandbox. "
            "Reponds avec une liste structuree de problemes trouves, "
            "classes par severite (CRITIQUE / MAJEUR / MINEUR). "
            "Si un probleme necessite un specialiste, ajoute une ligne: "
            "ROUTE:security pour les failles de securite, "
            "ROUTE:tester pour les cas limites non testes. "
            "Si le code est acceptable, reponds exactement: APPROVED"
        ),
        "temperature": 0.2,
        "max_tokens": 2048,
        "sandbox_access": "read_only",
        "tools": ["shell"],  # read-only commands only
    },
    AgentRole.CRITIC: {
        "name": "Critic Agent",
        "system_prompt": (
            "Tu es un auditeur hostile et impartial. "
            "Ton role est de chercher ACTIVEMENT les failles dans la reponse fournie : "
            "1. Hallucinations : affirmations non fondees ou inventees "
            "2. Erreurs logiques : raisonnements invalides, contradictions "
            "3. Failles de securite : injections, fuites de donnees, SSRF "
            "4. Cas limites : inputs vides, null, tres grands, caracteres speciaux "
            "5. Omissions : exigences du cahier des charges ignorees "
            "Sois impitoyable. Si tout est correct, reponds exactement: VALIDE. "
            "Sinon, liste CHAQUE probleme avec [ERREUR], [SECURITE] ou [OMISSION]."
        ),
        "temperature": 0.1,  # Tres deterministe pour la critique
        "max_tokens": 2048,
        "sandbox_access": "none",
        "tools": [],
    },
    AgentRole.RESEARCHER: {
        "name": "Researcher Agent",
        "system_prompt": (
            "Tu es un agent de recherche specialise. "
            "Tu analyses les questions, decompose les problemes, "
            "et fournis des analyses detaillees avec sources. "
            "Tu n'executes pas de code, tu fournis de l'information."
        ),
        "temperature": 0.5,
        "max_tokens": 3072,
        "sandbox_access": "none",
        "tools": [],
    },
    AgentRole.PLANNER: {
        "name": "Planner Agent",
        "system_prompt": (
            "Tu es un architecte logiciel. "
            "Tu decompose les taches complexes en etapes claires. "
            "Tu crees des plans d'action structures avec des criteres de validation. "
            "Tu identifies les risques et dependances."
        ),
        "temperature": 0.4,
        "max_tokens": 2048,
        "sandbox_access": "none",
        "tools": [],
    },
    AgentRole.TESTER: {
        "name": "Tester Agent",
        "system_prompt": (
            "Tu es un expert en tests logiciels. "
            "Tu ecris des tests unitaires et d'integration exhaustifs. "
            "Tu couvres les cas limites, les erreurs, et les cas nominaux. "
            "Tu utilises pytest et les meilleures pratiques de test Python."
        ),
        "temperature": 0.3,
        "max_tokens": 4096,
        "sandbox_access": "read_write",
        "tools": ["python", "shell"],
    },
    AgentRole.SECURITY: {
        "name": "Security Agent",
        "system_prompt": (
            "Tu es un expert en securite applicative (OWASP Top 10, SANS). "
            "Tu analyses le code pour detecter : "
            "1. Injections (SQL, commande, LDAP, XSS, SSTI) "
            "2. Fuites de secrets (cles API, tokens dans le code) "
            "3. SSRF et acces reseau non controle "
            "4. Deserialisation non securisee "
            "5. Controle d'acces insuffisant "
            "Reponds avec un rapport structure: "
            "[VULN-ID] Severite | Description | Ligne(s) | Remediation proposee. "
            "Si aucune faille, reponds exactement: SECURE"
        ),
        "temperature": 0.1,
        "max_tokens": 2048,
        "sandbox_access": "read_only",
        "tools": ["shell"],
    },
}

# Mapping for ROUTE: directives from Reviewer
ROUTE_MAPPING: dict[str, str] = {
    "security": AgentRole.SECURITY,
    "tester": AgentRole.TESTER,
    "researcher": AgentRole.RESEARCHER,
}


@dataclass
class SwarmAgent:
    """An active specialized agent instance in the swarm."""
    id: str = field(default_factory=lambda: f"swarm_{uuid.uuid4().hex[:8]}")
    role: str = AgentRole.CODER
    profile: dict = field(default_factory=dict)
    status: str = "idle"  # idle, working, completed, failed
    task: str = ""
    result: str = ""
    review_feedback: str = ""
    iteration: int = 0


@dataclass
class SwarmResult:
    """Result of a swarm execution cycle."""
    success: bool = False
    code: str = ""
    review: str = ""
    critic_verdict: str = ""
    final_output: str = ""
    iterations: int = 0
    validated: bool = False
    agents_used: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


class SwarmOrchestrator:
    """
    Manages a swarm of specialized sub-agents.

    Implements the Coder-Reviewer loop:
    1. Planner decomposes the task (optional)
    2. Coder generates code
    3. Reviewer critiques it
    4. Coder corrects based on feedback
    5. Orchestrator validates and returns

    Max iterations prevents infinite loops.
    """

    # Token-safe threshold: compress feedback when it exceeds this char count
    _FEEDBACK_COMPRESS_THRESHOLD = 3000

    def __init__(self, brain: "AgentBrain", ws_manager=None):
        self.settings = get_settings()
        self.brain = brain
        self._ws_manager = ws_manager
        self._max_iterations = self.settings.get("agent.swarm.max_iterations", 3)
        self._enabled = self.settings.get("agent.swarm.enabled", True)
        self._active_agents: dict[str, SwarmAgent] = {}
        # Human hint injection buffer (set via inject_hint)
        self._pending_hint: str = ""
        # Dry Run: compile-check code before passing to Reviewer
        self._dry_run_enabled = self.settings.get("agent.swarm.dry_run", True)

    async def execute_swarm(
        self,
        task: str,
        roles: list[str] = None,
        session_id: str = "",
    ) -> SwarmResult:
        """
        Execute a task using the swarm of specialized agents.

        Args:
            task: The task description
            roles: Which agent roles to involve (default: coder + reviewer)
            session_id: Session for context

        Returns:
            SwarmResult with code, review, and final output
        """
        if not self._enabled:
            # Fallback to single agent
            result = await self.brain.generate(
                messages=[{"role": "user", "content": task}],
            )
            return SwarmResult(
                success=True,
                final_output=result.get("content", ""),
                agents_used=["default"],
            )

        roles = roles or [AgentRole.CODER, AgentRole.REVIEWER]
        swarm_result = SwarmResult()
        trace = []

        logger.info(f"Swarm started: roles={roles}, task='{task[:80]}...'")

        # Phase 0: Generate Repository Map for context awareness
        repo_map = ""
        workspace = self.settings.get("sandbox.workspace_path", "/workspace")
        try:
            repo_map = generate_repo_map(workspace, max_chars=4000)
            if repo_map and len(repo_map) > 50:
                trace.append({"phase": "repo_map", "chars": len(repo_map)})
                logger.info(f"repo_map generated: {len(repo_map)} chars")
        except Exception as e:
            logger.debug(f"repo_map generation skipped: {e}")

        # Phase 1: Planning (if planner is included)
        if AgentRole.PLANNER in roles:
            planner_input = task
            if repo_map:
                planner_input = (
                    f"[CARTE DU PROJET]\n{repo_map}\n[/CARTE]\n\n{task}"
                )
            plan = await self._run_agent(AgentRole.PLANNER, planner_input)
            trace.append({"phase": "planning", "output": plan})
            swarm_result.agents_used.append(AgentRole.PLANNER)
            # Enrich task with plan
            task = f"Plan:\n{plan}\n\nOriginal task:\n{task}"

        # Phase 2: Coder-Reviewer Loop
        current_code = ""
        review_feedback = ""

        for iteration in range(1, self._max_iterations + 1):
            swarm_result.iterations = iteration

            # Coder phase
            if AgentRole.CODER in roles:
                # Inject pending human hint if available
                hint_block = ""
                if self._pending_hint:
                    hint_block = (
                        f"\n\n[MESSAGE URGENT DE L'UTILISATEUR]\n"
                        f"{self._pending_hint}\n"
                        f"[FIN DU MESSAGE]\n"
                    )
                    self._pending_hint = ""

                if review_feedback:
                    # Fading Memory: compress long feedback to avoid token overflow
                    effective_feedback = review_feedback
                    if (
                        iteration > 2
                        and len(review_feedback) > self._FEEDBACK_COMPRESS_THRESHOLD
                    ):
                        effective_feedback = await self._compress_feedback(
                            review_feedback, iteration
                        )
                        trace.append({
                            "phase": "feedback_compression",
                            "iteration": iteration,
                            "original_len": len(review_feedback),
                            "compressed_len": len(effective_feedback),
                        })

                    coder_task = (
                        f"Tache originale:\n{task}\n\n"
                        f"Code precedent:\n```python\n{current_code}\n```\n\n"
                        f"Feedback du Reviewer (iteration {iteration}):\n{effective_feedback}"
                        f"{hint_block}\n\n"
                        f"Corrige le code en tenant compte de TOUS les points souleves."
                    )
                else:
                    # First iteration: inject repo map for context awareness
                    map_block = ""
                    if repo_map and iteration == 1:
                        map_block = f"\n\n[CARTE DU PROJET]\n{repo_map}\n[/CARTE]"
                    coder_task = task + map_block + hint_block

                current_code = await self._run_agent(AgentRole.CODER, coder_task)
                swarm_result.code = current_code
                if AgentRole.CODER not in swarm_result.agents_used:
                    swarm_result.agents_used.append(AgentRole.CODER)

                trace.append({
                    "phase": "coding",
                    "iteration": iteration,
                    "output": current_code[:1000],
                })

            # Dry Run phase: py_compile before review
            dry_run_report = ""
            if current_code and self._dry_run_enabled:
                dry_run_report = await self._dry_run_compile(current_code)
                if dry_run_report:
                    trace.append({
                        "phase": "dry_run",
                        "iteration": iteration,
                        "output": dry_run_report[:500],
                    })
                    logger.info(f"Dry run: {dry_run_report[:120]}")

            # Reviewer phase
            if AgentRole.REVIEWER in roles and current_code:
                dry_run_block = ""
                if dry_run_report:
                    dry_run_block = (
                        f"\n\n[RESULTAT DRY RUN (py_compile)]\n"
                        f"{dry_run_report}\n[/DRY RUN]\n"
                    )
                review_task = (
                    f"Analyse ce code Python pour la tache suivante:\n"
                    f"Tache: {task[:500]}\n\n"
                    f"Code a analyser:\n```python\n{current_code}\n```"
                    f"{dry_run_block}\n\n"
                    f"Liste tous les problemes trouves. "
                    f"Si le code est acceptable, reponds exactement: APPROVED"
                )

                review_feedback = await self._run_agent(AgentRole.REVIEWER, review_task)
                swarm_result.review = review_feedback
                if AgentRole.REVIEWER not in swarm_result.agents_used:
                    swarm_result.agents_used.append(AgentRole.REVIEWER)

                trace.append({
                    "phase": "review",
                    "iteration": iteration,
                    "output": review_feedback[:1000],
                })

                # Check if approved
                if "APPROVED" in review_feedback.upper():
                    logger.info(f"Swarm: Code APPROVED at iteration {iteration}")
                    break

                # Dynamic routing: delegate to specialists if reviewer requests it
                routed_roles = self._parse_routing(review_feedback)
                for routed_role in routed_roles:
                    if routed_role not in swarm_result.agents_used:
                        swarm_result.agents_used.append(routed_role)

                    specialist_task = (
                        f"Le Reviewer a identifie des problemes dans ce code "
                        f"qui necessitent ton expertise.\n\n"
                        f"Code:\n```python\n{current_code}\n```\n\n"
                        f"Feedback du Reviewer:\n{review_feedback}\n\n"
                        f"Analyse et fournis un rapport detaille."
                    )
                    specialist_output = await self._run_agent(routed_role, specialist_task)
                    trace.append({
                        "phase": f"routed_{routed_role}",
                        "iteration": iteration,
                        "output": specialist_output[:1000],
                    })
                    # Enrich review feedback with specialist findings
                    review_feedback += f"\n\n[{routed_role.upper()} REPORT]\n{specialist_output}"

                logger.info(
                    f"Swarm: Review iteration {iteration} - corrections needed"
                    + (f" (routed to: {routed_roles})" if routed_roles else "")
                )

            # Tester phase (optional)
            if AgentRole.TESTER in roles and current_code:
                test_task = (
                    f"Ecris des tests pour ce code:\n```python\n{current_code}\n```"
                )
                test_output = await self._run_agent(AgentRole.TESTER, test_task)
                trace.append({
                    "phase": "testing",
                    "iteration": iteration,
                    "output": test_output[:1000],
                })
                if AgentRole.TESTER not in swarm_result.agents_used:
                    swarm_result.agents_used.append(AgentRole.TESTER)

        # Phase 3: Critic validation (final gate before returning to user)
        if AgentRole.CRITIC in roles and (current_code or review_feedback):
            output_to_validate = current_code or review_feedback
            critic_task = (
                f"Voici la reponse finale produite par l'essaim d'agents pour la tache suivante.\n\n"
                f"TACHE ORIGINALE:\n{task[:500]}\n\n"
                f"REPONSE A VALIDER:\n{output_to_validate}\n\n"
                f"Analyse cette reponse en tant qu'auditeur hostile. "
                f"Cherche les hallucinations, erreurs logiques, failles de securite, "
                f"cas limites non geres, et omissions. "
                f"Si tout est correct, reponds exactement: VALIDE"
            )

            critic_verdict = await self._run_agent(AgentRole.CRITIC, critic_task)
            swarm_result.critic_verdict = critic_verdict
            if AgentRole.CRITIC not in swarm_result.agents_used:
                swarm_result.agents_used.append(AgentRole.CRITIC)

            trace.append({
                "phase": "critic_validation",
                "output": critic_verdict[:1000],
                "validated": "VALIDE" in critic_verdict.upper(),
            })

            swarm_result.validated = "VALIDE" in critic_verdict.upper()

            if not swarm_result.validated:
                logger.warning(
                    f"Swarm: Critic REJECTED the output. Issues found."
                )
            else:
                logger.info("Swarm: Critic VALIDATED the output.")

        # Finalize
        swarm_result.success = True
        swarm_result.final_output = current_code or review_feedback
        swarm_result.trace = trace

        logger.info(
            f"Swarm completed: {swarm_result.iterations} iterations, "
            f"agents={swarm_result.agents_used}, validated={swarm_result.validated}"
        )
        return swarm_result

    async def _compress_feedback(self, feedback: str, iteration: int) -> str:
        """
        Fading Memory: summarize accumulated feedback to prevent token overflow.

        Instead of passing raw multi-iteration feedback to the Coder, we ask
        the LLM to produce a concise summary of all issues found so far.
        """
        compress_prompt = (
            f"Tu es un assistant de synthese. Voici le feedback accumule de {iteration} "
            f"iterations de revue de code. Resume-le en un paragraphe concis "
            f"qui capture TOUS les problemes encore non resolus. "
            f"Ne perds aucune information critique.\n\n"
            f"Feedback brut:\n{feedback[:6000]}\n\n"
            f"Resume concis:"
        )
        try:
            result = await self.brain.generate(
                messages=[{"role": "user", "content": compress_prompt}],
                max_tokens=800,
                temperature=0.1,
            )
            compressed = result.get("content", "").strip()
            if compressed and len(compressed) < len(feedback):
                logger.info(
                    f"Fading Memory: compressed feedback {len(feedback)} -> {len(compressed)} chars"
                )
                return compressed
        except Exception as e:
            logger.warning(f"Fading Memory compression failed: {e}")

        # Fallback: hard-truncate to keep the tail (most recent issues)
        return feedback[-self._FEEDBACK_COMPRESS_THRESHOLD:]

    async def _dry_run_compile(self, code: str) -> str:
        """
        Dry Run: attempt py_compile on the code to catch syntax errors early.

        Returns an empty string if compilation succeeds, or a diagnostic
        string describing the error(s) found.
        """
        import tempfile
        import py_compile
        import os

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="dryrun_")
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)

            py_compile.compile(tmp_path, doraise=True)
            return ""  # Success: no errors
        except py_compile.PyCompileError as e:
            # Extract the meaningful part of the error
            msg = str(e)
            # Try to rewrite temp path to something readable
            if tmp_path:
                msg = msg.replace(tmp_path, "<code>")
            return f"ERREUR DE COMPILATION:\n{msg}"
        except Exception as e:
            return f"ERREUR DRY RUN: {e}"
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def inject_hint(self, hint: str):
        """Inject a human hint to be consumed by the next Coder iteration."""
        self._pending_hint = hint
        logger.info(f"Human hint injected: '{hint[:80]}...'")

    @staticmethod
    def _parse_routing(review_output: str) -> list[str]:
        """Parse ROUTE: directives from reviewer output and return agent roles."""
        routed = []
        for line in review_output.splitlines():
            line_stripped = line.strip().upper()
            if line_stripped.startswith("ROUTE:"):
                target = line_stripped.split(":", 1)[1].strip().lower()
                mapped_role = ROUTE_MAPPING.get(target)
                if mapped_role and mapped_role not in routed:
                    routed.append(mapped_role)
        return routed

    async def _run_agent(self, role: str, task: str) -> str:
        """Run a single specialized agent with its profile."""
        profile = AGENT_PROFILES.get(role, AGENT_PROFILES[AgentRole.CODER])

        agent = SwarmAgent(role=role, profile=profile, task=task, status="working")
        self._active_agents[agent.id] = agent

        # Notify UI: agent spawned
        await self._emit_agent_event("agent_spawned", agent)

        try:
            messages = [
                {"role": "user", "content": task},
            ]

            result = await self.brain.generate(
                messages=messages,
                memory_context=f"[System Profile: {profile['system_prompt']}]",
                temperature=profile.get("temperature", 0.5),
                max_tokens=profile.get("max_tokens", 2048),
            )

            agent.result = result.get("content", "")
            agent.status = "completed"

            # Notify UI: agent completed
            await self._emit_agent_event("agent_completed", agent)
            return agent.result

        except Exception as e:
            logger.error(f"Swarm agent {role} failed: {e}")
            agent.status = "failed"
            await self._emit_agent_event("agent_failed", agent)
            return f"[Agent {role} error: {str(e)}]"

        finally:
            self._active_agents.pop(agent.id, None)

    async def _emit_agent_event(self, event_type: str, agent: SwarmAgent):
        """Emit a swarm agent lifecycle event via WebSocket."""
        if not self._ws_manager:
            return
        try:
            await self._ws_manager.broadcast({
                "type": event_type,
                "agent_id": agent.id,
                "role": agent.role,
                "status": agent.status,
                "task_preview": agent.task[:100],
            })
        except Exception:
            pass  # Non-critical, don't break the swarm

    def get_active_agents(self) -> list[dict]:
        """List currently active swarm agents."""
        return [
            {
                "id": a.id,
                "role": a.role,
                "status": a.status,
                "task": a.task[:100],
            }
            for a in self._active_agents.values()
        ]

    def get_available_profiles(self) -> dict:
        """List all available agent profiles."""
        return {
            role: {
                "name": profile["name"],
                "sandbox_access": profile["sandbox_access"],
                "tools": profile["tools"],
            }
            for role, profile in AGENT_PROFILES.items()
        }
