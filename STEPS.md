module purge
module load cuda-12.8.1-gcc-12.1.0
module load mamba/latest
source activate venv
module load ollama/0.20.4
ollama-start
apptainer run --writable-tmpfs --bind "$PWD/data/neo4j_data:/data" --bind "$PWD/data/neo4j_logs:/logs" --env NEO4J_AUTH=neo4j/clickless123 /packages/apps/simg/neo4j_5.15.0-ubi8.sif &

#### Neo4j debug commands to check if it is already running
curl -s http://localhost:7474 | head
pgrep -af neo4j
pgrep -af apptainer

### Check if Ollama is serving
pgrep -af ollama

#### First time ollama-setup
ollama pull mistral:7b
ollama pull llama3.2-vision:11b