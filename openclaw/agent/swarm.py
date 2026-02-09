"""
Swarm Mode - Specialized Sub-Agent System.

Instead of a single generalist agent, the Orchestrator can spawn
specialized agent instances with tailored system prompts and permissions:

- Coder Agent: Expert Python coder with full sandbox R/W access.
- Reviewer Agent: Security reviewer with read-only sandbox access.
- Researcher Agent: Focused on information gathering and analysis.

Flow: Coder writes code -> Reviewer critiques -> Coder corrects -> Orchestrator validates.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from openclaw.config.settings import get_settings

if TYPE_CHECKING:
    from openclaw.agent.brain import AgentBrain

logger = logging.getLogger("openclaw.agent.swarm")


class AgentRole(str, Enum):
    """Predefined specialized agent roles."""
    CODER = "coder"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    TESTER = "tester"


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
        "tools": ["python", "shell", "write_file"],
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
            "classes par severite (CRITIQUE / MAJEUR / MINEUR)."
        ),
        "temperature": 0.2,
        "max_tokens": 2048,
        "sandbox_access": "read_only",
        "tools": ["shell"],  # read-only commands only
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
    final_output: str = ""
    iterations: int = 0
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

    def __init__(self, brain: "AgentBrain"):
        self.settings = get_settings()
        self.brain = brain
        self._max_iterations = self.settings.get("agent.swarm.max_iterations", 3)
        self._enabled = self.settings.get("agent.swarm.enabled", True)
        self._active_agents: dict[str, SwarmAgent] = {}

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

        # Phase 1: Planning (if planner is included)
        if AgentRole.PLANNER in roles:
            plan = await self._run_agent(AgentRole.PLANNER, task)
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
                if review_feedback:
                    coder_task = (
                        f"Tache originale:\n{task}\n\n"
                        f"Code precedent:\n```python\n{current_code}\n```\n\n"
                        f"Feedback du Reviewer (iteration {iteration}):\n{review_feedback}\n\n"
                        f"Corrige le code en tenant compte de TOUS les points souleves."
                    )
                else:
                    coder_task = task

                current_code = await self._run_agent(AgentRole.CODER, coder_task)
                swarm_result.code = current_code
                if AgentRole.CODER not in swarm_result.agents_used:
                    swarm_result.agents_used.append(AgentRole.CODER)

                trace.append({
                    "phase": "coding",
                    "iteration": iteration,
                    "output": current_code[:1000],
                })

            # Reviewer phase
            if AgentRole.REVIEWER in roles and current_code:
                review_task = (
                    f"Analyse ce code Python pour la tache suivante:\n"
                    f"Tache: {task[:500]}\n\n"
                    f"Code a analyser:\n```python\n{current_code}\n```\n\n"
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

                logger.info(
                    f"Swarm: Review iteration {iteration} - corrections needed"
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

        # Finalize
        swarm_result.success = True
        swarm_result.final_output = current_code or review_feedback
        swarm_result.trace = trace

        logger.info(
            f"Swarm completed: {swarm_result.iterations} iterations, "
            f"agents={swarm_result.agents_used}"
        )
        return swarm_result

    async def _run_agent(self, role: str, task: str) -> str:
        """Run a single specialized agent with its profile."""
        profile = AGENT_PROFILES.get(role, AGENT_PROFILES[AgentRole.CODER])

        agent = SwarmAgent(role=role, profile=profile, task=task, status="working")
        self._active_agents[agent.id] = agent

        try:
            # Build messages with specialized system prompt
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
            return agent.result

        except Exception as e:
            logger.error(f"Swarm agent {role} failed: {e}")
            agent.status = "failed"
            return f"[Agent {role} error: {str(e)}]"

        finally:
            self._active_agents.pop(agent.id, None)

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
