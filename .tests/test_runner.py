import unittest
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules you want to test
import search_manager

# Define the test suite
suite = unittest.TestSuite()

# Add tests for each module
suite.addTest(unittest.makeSuite(search_manager.TestSearchManager))

# Run the tests
runner = unittest.TextTestRunner()
result = runner.run(suite)

# Print the test results
print(f"Tests run: {result.testsRun}")
print(f"Errors: {len(result.errors)}")
print(f"Failures: {len(result.failures)}")

# Exit with a non-zero code if there were errors or failures
if result.errors or result.failures:
    sys.exit(1)
