# Contributing

Thank you for your interest in contributing to this project!

## Running Tests

The test suite uses pytest. To run all tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_derivatives.py

# Run with coverage
pytest --cov=optical_flow tests/
```

## Test Structure

Tests are located in the `tests/` directory:
- `test_*.py` - Test files for each module
- `conftest.py` - Shared fixtures and test configuration

The test suite includes:
- Unit tests for core functionality (derivatives, sparse operations, etc.)
- Integration tests for optical flow methods (HS, BA, Classic+NL)
- I/O tests for .flo file reading/writing
- Metric computation tests

## Adding New Tests

When adding new features, please add corresponding tests:

1. Create or update test file in `tests/`
2. Use pytest fixtures from `conftest.py` when possible
3. Follow existing test patterns
4. Ensure tests pass before submitting PR

Example test structure:
```python
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.shape == expected_shape
    np.testing.assert_allclose(result, expected, atol=1e-6)
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where helpful
- Document functions with docstrings
- Keep functions focused and testable

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests to ensure everything works:
   ```bash
   pytest
   ```
5. Make your changes
6. Add tests for new functionality
7. Ensure all tests pass
8. Submit a pull request

## Performance Considerations

If you're adding computationally intensive code:
- Profile your code to identify bottlenecks
- Consider vectorization and efficient NumPy operations
- Maintain numerical accuracy

## Questions?

Feel free to open an issue for questions or discussions about contributions.
