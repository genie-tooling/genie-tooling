# Tutorial: Custom KeyProvider (E12)

This tutorial corresponds to the example file `examples/E12_custom_key_provider_example.py`.

It demonstrates how to implement and use a custom `KeyProvider` class. This is useful for integrating with custom secret management systems (like HashiCorp Vault, AWS Secrets Manager, etc.) instead of relying on environment variables. It shows how to:
- Create a class that implements the `KeyProvider` protocol.
- Implement the `async def get_key(...)` method.
- Pass an instance of your custom provider to `Genie.create()`.

## Example Code

--8<-- "examples/E12_custom_key_provider_example.py"
