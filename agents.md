# Agent Development Guide: Python & OpenAI Expert Agent

## Overview

This guide provides comprehensive instructions for crafting an expert-level AI agent that specializes in Python development and OpenAI API/SDK integration. The agent is designed with a strong emphasis on type annotations and Pydantic models for robust, maintainable, and production-ready code.

## Core Principles

### 1. Type Safety First
- **Always use type annotations** for function parameters, return types, and class attributes
- Leverage Python 3.9+ type hints including `Union`, `Optional`, `List`, `Dict`, etc.
- Use `typing_extensions` for advanced type features when needed

### 2. Pydantic for Data Validation
- **Use Pydantic models** for all data structures, API requests/responses, and configuration
- Implement custom validators for complex business logic
- Leverage Pydantic's automatic JSON serialization/deserialization

### 3. OpenAI SDK Best Practices
- Use the latest OpenAI Python SDK (v1.x)
- Implement proper error handling and retry logic
- Structure prompts and responses using Pydantic models

## Agent Architecture

### Base Agent Structure

```python
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from openai import OpenAI
import asyncio
from datetime import datetime

class AgentConfig(BaseModel):
    """Configuration for the AI agent."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="OpenAI model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    timeout: int = Field(default=30, gt=0)

    @validator('model')
    def validate_model(cls, v: str) -> str:
        allowed_models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']
        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v

class Message(BaseModel):
    """Represents a message in the conversation."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('role')
    def validate_role(cls, v: str) -> str:
        if v not in ['system', 'user', 'assistant']:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        return v

class AgentResponse(BaseModel):
    """Structured response from the agent."""
    content: str = Field(..., description="Response content")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning process")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")

class PythonExpertAgent:
    """Expert Python and OpenAI agent with type safety and Pydantic integration."""

    def __init__(self, config: AgentConfig) -> None:
        self.config: AgentConfig = config
        self.client: OpenAI = OpenAI(api_key=config.api_key)
        self.conversation_history: List[Message] = []

    async def process_request(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process a user request and return a structured response."""
        # Implementation here
        pass
```

### Key Components to Implement

#### 1. Tool Integration with Pydantic

```python
class ToolParameter(BaseModel):
    """Define tool parameters with validation."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True)
    default: Optional[Any] = Field(None)

class Tool(BaseModel):
    """Tool definition for function calling."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: List[ToolParameter] = Field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        # Implementation here
        pass

class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
```

#### 2. Error Handling and Validation

```python
from pydantic import ValidationError
from openai import OpenAIError
import logging

class AgentError(Exception):
    """Base exception for agent errors."""
    pass

class ValidationError(AgentError):
    """Raised when input validation fails."""
    pass

class OpenAIError(AgentError):
    """Raised when OpenAI API calls fail."""
    pass

def handle_errors(func):
    """Decorator for comprehensive error handling."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logging.error(f"Validation error: {e}")
            raise ValidationError(f"Input validation failed: {e}")
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            raise OpenAIError(f"OpenAI API call failed: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise AgentError(f"Unexpected error occurred: {e}")
    return wrapper
```

#### 3. Advanced Pydantic Features

```python
from pydantic import BaseModel, Field, root_validator, validator
from typing import Union, Literal
from enum import Enum

class TaskType(str, Enum):
    """Enumeration of supported task types."""
    CODE_REVIEW = "code_review"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"

class CodeContext(BaseModel):
    """Context for code-related tasks."""
    language: str = Field(default="python", description="Programming language")
    framework: Optional[str] = Field(None, description="Framework being used")
    libraries: List[str] = Field(default_factory=list, description="Required libraries")
    complexity: Literal["simple", "medium", "complex"] = Field(default="medium")

class TaskRequest(BaseModel):
    """Structured task request."""
    task_type: TaskType = Field(..., description="Type of task to perform")
    description: str = Field(..., min_length=10, description="Task description")
    code_context: Optional[CodeContext] = Field(None)
    constraints: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)

    @root_validator
    def validate_code_context(cls, values):
        """Ensure code context is provided for code-related tasks."""
        task_type = values.get('task_type')
        code_context = values.get('code_context')

        if task_type in [TaskType.CODE_REVIEW, TaskType.CODE_GENERATION, TaskType.DEBUGGING]:
            if not code_context:
                raise ValueError(f"Code context required for {task_type}")

        return values
```

## Implementation Guidelines

### 1. OpenAI API Integration

```python
async def call_openai_api(
    self,
    messages: List[Message],
    tools: Optional[List[Tool]] = None
) -> AgentResponse:
    """Make a call to OpenAI API with proper error handling."""

    try:
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Prepare API call parameters
        api_params = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = [tool.to_openai_format() for tool in tools]

        # Make the API call
        response = await self.client.chat.completions.create(**api_params)

        # Extract response data
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return AgentResponse(
            content=content,
            tokens_used=tokens_used,
            confidence=0.9,  # Calculate based on response
            metadata={"model": self.config.model}
        )

    except Exception as e:
        raise OpenAIError(f"API call failed: {e}")
```

### 2. Advanced Type Annotations

```python
from typing import TypeVar, Generic, Callable, Awaitable, Protocol

T = TypeVar('T')
P = TypeVar('P', bound=BaseModel)

class ProcessorProtocol(Protocol):
    """Protocol for request processors."""

    async def process(self, request: BaseModel) -> BaseModel:
        """Process a request and return a response."""
        ...

class AgentProcessor(Generic[T]):
    """Generic processor for different request types."""

    def __init__(
        self,
        processor_func: Callable[[T], Awaitable[AgentResponse]]
    ) -> None:
        self.processor_func = processor_func

    async def process(self, request: T) -> AgentResponse:
        """Process a typed request."""
        return await self.processor_func(request)

# Usage example
def create_code_processor() -> AgentProcessor[TaskRequest]:
    """Create a processor for code-related tasks."""

    async def process_code_task(request: TaskRequest) -> AgentResponse:
        # Implementation here
        pass

    return AgentProcessor(process_code_task)
```

### 3. Configuration Management

```python
from pydantic import BaseSettings
from typing import Optional
import os

class AgentSettings(BaseSettings):
    """Application settings with environment variable support."""

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    default_model: str = Field(default="gpt-4", env="DEFAULT_MODEL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
settings = AgentSettings()
agent_config = AgentConfig(
    api_key=settings.openai_api_key,
    model=settings.default_model
)
```

## Best Practices

### 1. Type Safety
- Use `mypy` for static type checking
- Add type annotations to all functions and methods
- Use `typing_extensions` for advanced features

### 2. Pydantic Usage
- Create models for all data structures
- Use validators for complex validation logic
- Leverage automatic JSON serialization
- Use `Field()` for documentation and constraints

### 3. Error Handling
- Create custom exception hierarchies
- Use decorators for consistent error handling
- Log errors with appropriate context
- Provide meaningful error messages

### 4. Testing
- Write unit tests for all Pydantic models
- Test validation logic thoroughly
- Mock OpenAI API calls in tests
- Use `pytest` with async support

### 5. Documentation
- Document all classes and methods
- Use docstrings with type information
- Provide usage examples
- Keep documentation up to date

## Example Usage

```python
async def main():
    """Example usage of the Python Expert Agent."""

    # Load configuration
    settings = AgentSettings()
    config = AgentConfig(api_key=settings.openai_api_key)

    # Create agent
    agent = PythonExpertAgent(config)

    # Create a task request
    task = TaskRequest(
        task_type=TaskType.CODE_GENERATION,
        description="Create a Pydantic model for user management",
        code_context=CodeContext(
            language="python",
            libraries=["pydantic", "fastapi"]
        )
    )

    # Process the request
    response = await agent.process_request(
        user_input=task.description,
        context={"task": task.dict()}
    )

    print(f"Response: {response.content}")
    print(f"Confidence: {response.confidence}")
    print(f"Tokens used: {response.tokens_used}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

This guide provides a comprehensive framework for building expert-level Python and OpenAI agents with strong emphasis on type safety and Pydantic integration. The resulting agents will be robust, maintainable, and production-ready.

Key takeaways:
- Always use type annotations for better code quality
- Leverage Pydantic for data validation and serialization
- Implement proper error handling and logging
- Structure code for maintainability and testability
- Follow OpenAI SDK best practices for API integration

Remember to continuously test, document, and refine your agent implementation as requirements evolve.
