FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install uv
RUN pip install uv

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 8080

# Run the agent
CMD ["uv", "run", "app", "--host", "0.0.0.0"]
