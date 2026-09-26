#!/bin/bash
# Test script for margAI Docker deployment

set -e

echo "🧪 Testing margAI Docker Deployment"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if shared_net exists
echo -n "1. Checking shared_net network... "
if docker network inspect shared_net >/dev/null 2>&1; then
    echo -e "${GREEN}✓ exists${NC}"
else
    echo -e "${RED}✗ not found${NC}"
    echo "   Run: docker compose -f ~/docker/shared/docker-compose.yml up -d"
    exit 1
fi

# Test 2: Check if margAI container is running
echo -n "2. Checking margAI container... "
if docker ps | grep -q margai; then
    echo -e "${GREEN}✓ running${NC}"
else
    echo -e "${RED}✗ not running${NC}"
    echo "   Run: docker compose up -d"
    exit 1
fi

# Test 3: Check container health
echo -n "3. Checking container health... "
HEALTH=$(docker inspect margai --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-health-check")
if [ "$HEALTH" = "healthy" ]; then
    echo -e "${GREEN}✓ healthy${NC}"
elif [ "$HEALTH" = "starting" ]; then
    echo -e "${YELLOW}⚠ starting (wait a moment)${NC}"
else
    echo -e "${YELLOW}⚠ $HEALTH${NC}"
fi

# Test 4: Test API from host
echo -n "4. Testing API from host (localhost:8002)... "
if curl -sf http://localhost:8002/v1/models >/dev/null 2>&1; then
    echo -e "${GREEN}✓ accessible${NC}"
else
    echo -e "${RED}✗ not accessible${NC}"
    echo "   Check logs: docker logs margai"
    exit 1
fi

# Test 5: Test Windows Ollama connectivity from container
echo -n "5. Testing Ollama connectivity from margAI... "
if docker exec margai curl -sf http://host.docker.internal:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓ accessible${NC}"
else
    echo -e "${RED}✗ not accessible${NC}"
    echo "   Windows Ollama must listen on 0.0.0.0:11434"
    echo "   Check: OLLAMA_HOST=0.0.0.0:11434"
    echo "   Check: Windows Firewall allows TCP 11434"
fi

# Test 6: List available models
echo ""
echo "6. Available models:"
echo "-------------------"
MODELS=$(curl -s http://localhost:8002/v1/models | python3 -c "import sys, json; data = json.load(sys.stdin); print('\n'.join([m['id'] for m in data.get('data', [])]))" 2>/dev/null || echo "Could not fetch models")
if [ "$MODELS" = "Could not fetch models" ]; then
    echo -e "${RED}✗ Could not fetch models${NC}"
    echo "   Check logs: docker logs margai"
else
    echo "$MODELS" | while read -r model; do
        echo -e "  ${GREEN}✓${NC} $model"
    done
fi

# Test 7: Test from another container (if LibreChat is running)
echo ""
echo -n "7. Testing inter-container connectivity... "
if docker ps | grep -q LibreChat; then
    if docker exec LibreChat curl -sf http://margai:8002/v1/models >/dev/null 2>&1; then
        echo -e "${GREEN}✓ accessible from LibreChat${NC}"
    else
        echo -e "${RED}✗ not accessible from LibreChat${NC}"
        echo "   Both containers must be on shared_net"
    fi
else
    echo -e "${YELLOW}⚠ LibreChat not running (skipped)${NC}"
fi

echo ""
echo "===================================="
echo -e "${GREEN}✓ margAI Docker tests completed!${NC}"
echo ""
echo "Next steps:"
echo "  • View logs: docker logs margai"
echo "  • Test API: curl http://localhost:8002/v1/models"
echo "  • Start LibreChat: cd ~/docker/LibreChat && docker compose up -d"
echo "  • See guide: ~/codebox/margAI/DOCKER.md"
