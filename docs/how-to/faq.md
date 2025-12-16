# Frequently Asked Questions

## Why does the code use "falcon" when FutureHouse's service is called Edison?

FutureHouse originally offered a deep research tool called **Falcon**. They later retired that service and launched **Edison**, which provides essentially the same functionality.

Our code and environment variables (e.g., `FALCON_API_KEY`) retain the "falcon" naming for backward compatibility with existing configurations. If you see references to "edison.py" in older documentation or discussions, this is the historical context.

In practice:

- Use `FALCON_API_KEY` for your FutureHouse/Edison API key
- The provider is named `falcon` in CLI commands and configuration
- Under the hood, this connects to FutureHouse's Edison service
