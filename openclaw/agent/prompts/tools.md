# Outils Disponibles

## Outils Systeme (par defaut)

### `shell`
Execute une commande shell/terminal.
- **Arguments** : `command` (str), `timeout` (int, optionnel)
- **Usage** : Operations systeme, installation de paquets, git, etc.

### `read_file`
Lit le contenu d'un fichier.
- **Arguments** : `path` (str)
- **Usage** : Lire du code, des configurations, des documents

### `write_file`
Ecrit du contenu dans un fichier (cree ou ecrase).
- **Arguments** : `path` (str), `content` (str)
- **Usage** : Creer ou modifier des fichiers

### `search_files`
Recherche des fichiers par motif glob.
- **Arguments** : `path` (str), `pattern` (str)
- **Usage** : Trouver des fichiers dans l'arborescence

## Philosophie AgentZero

Ces 4 outils de base suffisent pour accomplir la plupart des taches.
Pour des besoins specifiques, l'agent peut :
1. Combiner les outils existants
2. Ecrire et executer des scripts
3. Creer de nouveaux skills dynamiquement

## Utilisation

Quand tu dois utiliser un outil :
1. Explique pourquoi tu l'utilises
2. Execute l'outil
3. Analyse le resultat
4. Continue ou ajuste selon le resultat
