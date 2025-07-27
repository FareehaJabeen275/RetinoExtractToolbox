import subprocess

def test_cli_runs():
    result = subprocess.run(['python', 'main.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()