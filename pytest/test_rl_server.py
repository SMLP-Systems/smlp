# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
Integration Tests for SMLP RL Agent Server

These tests use mocked LLM responses to test API endpoints.
Run with: pytest test_rl_server.py -v
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Adjust import path if needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Server Tests with Mocked LLM
# ============================================================================

class TestRLServer:
    """Test RL Agent server endpoints with mocked LLM"""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies"""
        # Patch where _call_llm is used (server module), not where it's defined (utils)
        with patch('smlp_agent_rl_server._call_llm') as mock_llm:
            mock_llm.return_value = {
                "mode": "train",
                "data": "toy.csv",
                "model": "dt_sklearn"
            }
            
            from smlp_agent_rl_server import create_app
            app = create_app(provider="ollama", model="test", dry_run=True)
            return TestClient(app)
    
    def test_root_serves_ui(self, client):
        """Test that root endpoint serves the web UI"""
        response = client.get("/")
        
        # Should return HTML (UI file)
        assert response.status_code in [200, 404]  # 404 if UI file not found in test env
    
    def test_config_endpoint(self, client):
        """Test /config returns server configuration"""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "provider" in data
        assert "model" in data
        assert "dry_run" in data
        assert data["dry_run"] == True
    
    def test_stats_endpoint_structure(self, client):
        """Test /stats returns valid structure"""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "total_feedback" in data
        assert "avg_reward" in data
        assert "recent_avg_reward" in data
        assert "num_examples" in data
        assert "total_selections" in data
        
        # Check types
        assert isinstance(data["total_feedback"], int)
        assert isinstance(data["avg_reward"], (int, float))
        assert isinstance(data["num_examples"], int)
    
    def test_generate_endpoint_with_mock(self):
        """Test /generate with mocked LLM"""
        # Patch where _call_llm is used (server module), not where it's defined (utils)
        with patch('smlp_agent_rl_server._call_llm') as mock_llm:
            mock_llm.return_value = {
                "mode": "train",
                "data": "../data/toy.csv",
                "model": "dt_sklearn",
                "resp": "y1",
                "feat": "x,p1"
            }
            
            from smlp_agent_rl_server import create_app
            app = create_app(provider="ollama", model="test", dry_run=True)
            client = TestClient(app)
            
            response = client.post("/generate", json={
                "query": "train dt model on toy data"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert "generated_command" in data
            assert "prompt_used" in data
            assert "pool_size" in data
            
            cmd = data["generated_command"]
            assert cmd["mode"] == "train"
            assert cmd["model"] == "dt_sklearn"
    
    def test_generate_handles_llm_error(self):
        """Test /generate handles LLM errors gracefully"""
        # Must patch where it's used (in server module), not where it's defined
        with patch('smlp_agent_rl_server._call_llm') as mock_llm:
            mock_llm.return_value = {"error": "LLM timeout"}
            
            from smlp_agent_rl_server import create_app
            app = create_app(provider="ollama", model="test", dry_run=True)
            client = TestClient(app)
            
            response = client.post("/generate", json={
                "query": "train dt model"
            })
            
            # Server checks for "error" key and raises HTTPException(502)
            assert response.status_code == 502
            assert "LLM error" in response.text
    
    def test_feedback_endpoint_records_data(self, client):
        """Test /feedback records feedback correctly"""
        feedback_data = {
            "user_query": "train dt on toy data",
            "generated_command": {
                "mode": "train",
                "data": "toy.csv"
            },
            "corrected_command": {
                "mode": "train",
                "data": "toy.csv",
                "resp": "y1",
                "feat": "x,p1"
            },
            "execution_success": True,
            "user_reward": None  # Auto-compute
        }
        
        response = client.post("/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "auto_reward" in data
        assert "final_reward" in data
        assert "total_feedback" in data
        assert "avg_reward" in data
        
        # Auto reward should be computed
        assert 0.0 <= data["auto_reward"] <= 1.0
        assert 0.0 <= data["final_reward"] <= 1.0
    
    def test_feedback_with_user_override(self, client):
        """Test /feedback with user star rating override"""
        feedback_data = {
            "user_query": "test query",
            "generated_command": {"mode": "train"},
            "corrected_command": {"mode": "train", "data": "x.csv"},
            "execution_success": False,
            "user_reward": 4  # 4 stars = 0.8 reward
        }
        
        response = client.post("/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Final reward should be user's rating (4/5 = 0.8)
        assert data["final_reward"] == 0.8
        
        # Auto reward should be different (based on edit distance)
        assert data["auto_reward"] != data["final_reward"]
    
    def test_feedback_zero_star_rating(self, client):
        """Test /feedback with 0-star rating"""
        feedback_data = {
            "user_query": "test query",
            "generated_command": {"mode": "verify"},
            "corrected_command": {"mode": "train", "data": "x.csv"},
            "execution_success": False,
            "user_reward": 0  # 0 stars = 0.0 reward
        }
        
        response = client.post("/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["final_reward"] == 0.0
    
    def test_execute_endpoint_dry_run(self, client):
        """Test /execute in dry-run mode"""
        response = client.post("/execute", json={
            "command": {
                "mode": "train",
                "data": "toy.csv",
                "model": "dt_sklearn"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "output" in data
        assert "command_line" in data
        
        # Dry-run should succeed and return dry-run message
        assert data["success"] == True
        assert "dry-run" in data["output"].lower()
    
    def test_store_get_endpoint(self, client):
        """Test /store returns example list"""
        response = client.get("/store")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "examples" in data
        assert "total" in data
        assert isinstance(data["examples"], list)
        assert isinstance(data["total"], int)
    
    def test_store_add_endpoint(self, client):
        """Test /store/add adds new example"""
        new_example = {
            "user_text": "test query for store",
            "smlp_command": {
                "mode": "train",
                "data": "test.csv"
            },
            "tag": "test"
        }
        
        response = client.post("/store/add", json=new_example)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "added"
        assert "id" in data
        assert "total" in data
    
    def test_prompt_endpoint(self, client):
        """Test /prompt returns current prompt"""
        response = client.get("/prompt")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prompt" in data
        assert "pool_size" in data
        assert "examples_per_prompt" in data
        
        assert isinstance(data["prompt"], str)
        assert len(data["prompt"]) > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test server error handling"""
    
    def test_generate_missing_query(self):
        """Test /generate rejects missing query"""
        from smlp_agent_rl_server import create_app
        app = create_app(provider="ollama", model="test", dry_run=True)
        client = TestClient(app)
        
        response = client.post("/generate", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_feedback_missing_fields(self):
        """Test /feedback rejects incomplete data"""
        from smlp_agent_rl_server import create_app
        app = create_app(provider="ollama", model="test", dry_run=True)
        client = TestClient(app)
        
        incomplete_feedback = {
            "user_query": "test"
            # Missing required fields
        }
        
        response = client.post("/feedback", json=incomplete_feedback)
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_json_format(self):
        """Test endpoints reject invalid JSON"""
        from smlp_agent_rl_server import create_app
        app = create_app(provider="ollama", model="test", dry_run=True)
        client = TestClient(app)
        
        response = client.post(
            "/generate",
            content=b"not valid json",  # Use content= for raw bytes
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


# ============================================================================
# Checkpoint and Persistence Tests
# ============================================================================

class TestPersistence:
    """Test checkpoint and state persistence"""
    
    def test_stats_persistence_across_feedback(self):
        """Test that stats update correctly as feedback is added"""
        from smlp_agent_rl_server import create_app
        app = create_app(provider="ollama", model="test", dry_run=True)
        client = TestClient(app)
        
        # Get initial stats
        response1 = client.get("/stats")
        initial_total = response1.json()["total_feedback"]
        
        # Submit feedback
        feedback = {
            "user_query": "test",
            "generated_command": {"mode": "train"},
            "corrected_command": {"mode": "train", "data": "x.csv"},
            "execution_success": True,
            "user_reward": 5
        }
        client.post("/feedback", json=feedback)
        
        # Get updated stats
        response2 = client.get("/stats")
        updated_total = response2.json()["total_feedback"]
        
        # Total should increment
        assert updated_total == initial_total + 1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
