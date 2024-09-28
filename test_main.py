from fastapi.testclient import TestClient
from main import app  # Use absolute import instead of relative

client = TestClient(app)

def test_dodge():
    response = client.post("/dodge")
    print(response.json())
    
    assert response.status_code == 200
    
    expected_grid = [
        ['.', 'd', 'd'],
        ['r', '*', '.'],
        ['.', '.', '.']
    ]

    print(response.json())
    
    assert response.json() == expected_grid