# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

"""
End-to-End Smoke Tests for SMLP RL Agent

These tests require the server to be running and Ollama to be loaded.
They are non-deterministic due to LLM involvement.

DO NOT run in CI/CD - these are for manual validation only.

Usage:
    1. Start the server in another terminal:
       python smlp_agent_rl_server.py --provider ollama --model deepseek-r1:1.5b --port 8000
    
    2. Pre-load the model:
       ollama run deepseek-r1:1.5b "test"
    
    3. Run this test:
       python test_e2e_manual.py
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"


def test_server_alive():
    """
    Test 1: Check that server is responding
    """
    print("\n" + "="*70)
    print("TEST 1: Server Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/config", timeout=5)
        assert response.status_code == 200
        
        config = response.json()
        print(f"✓ Server is alive")
        print(f"  Provider: {config['provider']}")
        print(f"  Model: {config['model']}")
        print(f"  Dry-run: {config['dry_run']}")
        
        return True
    except Exception as e:
        print(f"✗ Server health check failed: {e}")
        print(f"  Make sure server is running: python smlp_agent_rl_server.py --port 8000")
        return False


def test_stats_endpoint():
    """
    Test 2: Check stats endpoint returns valid data
    """
    print("\n" + "="*70)
    print("TEST 2: Stats Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        assert response.status_code == 200
        
        stats = response.json()
        print(f"✓ Stats endpoint working")
        print(f"  Total feedback: {stats['total_feedback']}")
        print(f"  Avg reward: {stats['avg_reward']:.3f}")
        print(f"  Pool size: {stats['num_examples']}")
        
        return True
    except Exception as e:
        print(f"✗ Stats endpoint failed: {e}")
        return False


def test_generate_command():
    """
    Test 3: Generate a command (may fail due to LLM)
    """
    print("\n" + "="*70)
    print("TEST 3: Command Generation (non-deterministic)")
    print("="*70)
    
    query = "train dt on toy data"
    print(f"Query: '{query}'")
    print(f"Calling LLM... (this may take 10-30 seconds)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"query": query},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            cmd = data["generated_command"]
            
            print(f"✓ Generate succeeded")
            print(f"  Generated command: {json.dumps(cmd, indent=2)}")
            return cmd
        else:
            print(f"⚠ Generate failed with status {response.status_code}")
            print(f"  This is expected if LLM returns invalid JSON")
            print(f"  Error: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"⚠ Generate timed out (LLM took too long)")
        print(f"  Try pre-loading model: ollama run deepseek-r1:1.5b 'test'")
        return None
    except Exception as e:
        print(f"⚠ Generate failed: {e}")
        return None


def test_feedback_submission(generated_cmd):
    """
    Test 4: Submit feedback for a generated command
    """
    print("\n" + "="*70)
    print("TEST 4: Feedback Submission")
    print("="*70)
    
    if generated_cmd is None:
        print("⊘ Skipping (no generated command from Test 3)")
        return False
    
    # Create corrected command (add a field)
    corrected = {**generated_cmd, "resp": "y1", "feat": "x,p1"}
    
    feedback = {
        "user_query": "train dt on toy data",
        "generated_command": generated_cmd,
        "corrected_command": corrected,
        "execution_success": False,  # Assume execution not run
        "user_reward": 3  # 3 stars = 0.6 reward
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/feedback",
            json=feedback,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"✓ Feedback submitted successfully")
        print(f"  Auto reward: {data['auto_reward']:.3f}")
        print(f"  Final reward: {data['final_reward']:.3f}")
        print(f"  Total feedback: {data['total_feedback']}")
        
        return True
    except Exception as e:
        print(f"✗ Feedback submission failed: {e}")
        return False


def test_store_operations():
    """
    Test 5: Test example store operations
    """
    print("\n" + "="*70)
    print("TEST 5: Example Store Operations")
    print("="*70)
    
    try:
        # Get current store
        response = requests.get(f"{BASE_URL}/store", timeout=5)
        assert response.status_code == 200
        
        initial_total = response.json()["total"]
        print(f"✓ Store has {initial_total} examples")
        
        # Add a test example
        new_example = {
            "user_text": "test smoke test query",
            "smlp_command": {
                "mode": "train",
                "data": "smoke_test.csv",
                "model": "dt_sklearn"
            },
            "tag": "smoke_test"
        }
        
        response = requests.post(
            f"{BASE_URL}/store/add",
            json=new_example,
            timeout=5
        )
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"✓ Added example to store")
        print(f"  New example ID: {data['id']}")
        print(f"  Total examples: {data['total']}")
        
        assert data['total'] == initial_total + 1
        
        return True
    except Exception as e:
        print(f"✗ Store operations failed: {e}")
        return False


def test_execute_dry_run():
    """
    Test 6: Test command execution (dry-run)
    """
    print("\n" + "="*70)
    print("TEST 6: Command Execution (dry-run)")
    print("="*70)
    
    test_command = {
        "mode": "train",
        "data": "toy.csv",
        "model": "dt_sklearn"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/execute",
            json={"command": test_command},
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"✓ Execute endpoint working")
        print(f"  Success: {data['success']}")
        print(f"  Output: {data['output'][:100]}...")
        
        # In dry-run mode, should always succeed
        if "dry-run" in data['output'].lower():
            print(f"  (Dry-run mode confirmed)")
        
        return True
    except Exception as e:
        print(f"✗ Execute failed: {e}")
        return False


def run_all_tests():
    """
    Run all smoke tests in sequence
    """
    print("\n" + "="*70)
    print("SMLP RL AGENT - END-TO-END SMOKE TESTS")
    print("="*70)
    print("\nWARNING: These tests are non-deterministic!")
    print("Some tests may fail due to LLM issues - that's expected.")
    print("\nPre-requisites:")
    print("  1. Server running: python smlp_agent_rl_server.py --port 8000")
    print("  2. Model loaded: ollama run deepseek-r1:1.5b 'test'")
    print("\n" + "="*70)
    
    results = {}
    
    # Test 1: Server health
    results["server_alive"] = test_server_alive()
    if not results["server_alive"]:
        print("\n✗ Server is not running - stopping tests")
        return results
    
    # Test 2: Stats endpoint
    results["stats"] = test_stats_endpoint()
    
    # Test 3: Generate command (may fail)
    generated_cmd = test_generate_command()
    results["generate"] = generated_cmd is not None
    
    # Test 4: Feedback (requires Test 3 to succeed)
    results["feedback"] = test_feedback_submission(generated_cmd)
    
    # Test 5: Store operations
    results["store"] = test_store_operations()
    
    # Test 6: Execute
    results["execute"] = test_execute_dry_run()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    elif passed >= total - 1:
        print("\n⚠ Most tests passed (LLM flakiness is expected)")
    else:
        print("\n⚠ Multiple tests failed - check server and Ollama")
    
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with error code if critical tests failed
    critical_tests = ["server_alive", "stats", "store", "execute"]
    critical_passed = all(results.get(t, False) for t in critical_tests)
    
    exit(0 if critical_passed else 1)
