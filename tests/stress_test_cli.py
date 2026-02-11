
import sys
import pexpect
import time

def run_stress_test():
    print("Starting stress test: 1000 lines paste...")
    # Generate 1000 lines of text
    huge_input = "\n".join([f"Line {i} of massive paste" for i in range(1000)])
    
    # Bracketed paste sequence + Enter to submit
    bracketed_input = f"\x1b[200~{huge_input}\x1b[201~\r"

    # Start nanobot agent
    child = pexpect.spawn(".venv/bin/python -m nanobot.cli.commands agent", encoding="utf-8", timeout=30)
    
    # Wait for prompt
    try:
        child.expect("You:", timeout=10)
        print("✅ Prompt detected")
        
        # Send massive input
        print(f"Sending {len(huge_input)} bytes...")
        child.send(bracketed_input)
        
        # Wait for thinking indicator
        child.expect("nanobot is thinking...", timeout=20)
        print("✅ Thinking indicator detected")
        
        # Wait for response (assuming default mocked response or error handling)
        # Note: If no API key, it might error, but input layer should still handle the paste cleanly
        # We look for the prompt to return
        child.expect("You:", timeout=20)
        print("✅ Returned to prompt after massive paste")
        
        # Send exit command
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=5)
        print("✅ Clean exit")
        
    except pexpect.exceptions.ExceptionPexpect as e:
        print(f"❌ Test Failed: {e}")
        print("Last output:")
        print(child.before)
        sys.exit(1)

if __name__ == "__main__":
    run_stress_test()
