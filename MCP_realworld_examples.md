# Real-World MCP Resources & Prompts: Invocation Examples
## Complete Guide with Actual Resources and Prompts (Not Just Tools)

---

## Table of Contents
1. [Understanding MCP Primitives](#understanding-mcp-primitives)
2. [Resources: Real-World Examples](#resources-real-world-examples)
3. [Prompts: Real-World Examples](#prompts-real-world-examples)
4. [GitHub MCP Server: Resources & Prompts](#github-mcp-server)
5. [Azure MCP Server: Resources & Prompts](#azure-mcp-server)
6. [Complete Workflow Examples](#complete-workflow-examples)

---

## Understanding MCP Primitives

### **The Three Core Interaction Types**

1. **Resources** (Application-Controlled)
   - Read-only data exposed via URIs
   - Provides context to LLMs
   - Client decides when to use them
   - Example: `file:///logs/app.log`, `database://schema/users`

2. **Prompts** (User-Controlled)
   - Pre-defined prompt templates with parameters
   - Exposed through slash commands or menus
   - User explicitly invokes them
   - Example: `/summarize`, `/code-review`

3. **Tools** (Model-Controlled)
   - Actions the AI can invoke
   - AI decides when to call them
   - Example: `create_issue`, `search_code`

---

## Resources: Real-World Examples

### **🔍 What Are Resources?**

Resources expose data through **URI-based addressing**. They are read-only and provide context to AI models.

---

### **Example 1: Static File Resource**

**Server Definition (Python):**
```python
from mcp import MCPResource

@mcp.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="file:///logs/app.log",
            name="Application Logs",
            description="Recent application log entries",
            mimeType="text/plain"
        )
    ]

@mcp.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    if str(uri) == "file:///logs/app.log":
        with open("/var/logs/app.log", "r") as f:
            return f.read()
    raise ValueError("Resource not found")
```

**Client Invocation (Claude Desktop):**
```
User: "Can you analyze the application logs?"

Claude internally:
1. Discovers resource via resources/list
2. Requests resource via resources/read
   Request: {"uri": "file:///logs/app.log"}
   Response: {
     "contents": [{
       "uri": "file:///logs/app.log",
       "mimeType": "text/plain",
       "text": "2025-01-15 ERROR: Database connection timeout..."
     }]
   }
3. Uses log content as context to analyze

Claude: "Based on the application logs, I see several database 
connection timeout errors starting at 14:32..."
```

**VS Code with MCP:**
```
# Reference resource in Copilot Chat
"Analyze #file:///logs/app.log for patterns"

# VS Code fetches the resource and includes it in context
```

---

### **Example 2: Dynamic Resource Templates**

**Server Definition (TypeScript):**
```typescript
// Resource Template with Parameters
server.setRequestHandler(
  ListResourceTemplatesRequestSchema, 
  async () => ({
    resourceTemplates: [{
      uriTemplate: "database://schema/{tableName}",
      name: "Database Schema",
      description: "Schema for any table in the database",
      mimeType: "application/json"
    }]
  })
);

server.setRequestHandler(
  ReadResourceRequestSchema, 
  async (request) => {
    const pattern = /^database:\/\/schema\/(.+)$/;
    const match = request.params.uri.match(pattern);
    
    if (match) {
      const tableName = decodeURIComponent(match[1]);
      const schema = await getTableSchema(tableName);
      
      return {
        contents: [{
          uri: request.params.uri,
          mimeType: "application/json",
          text: JSON.stringify(schema)
        }]
      };
    }
    
    throw new Error("Resource not found");
  }
);
```

**Client Invocation:**
```
User: "Show me the schema for the users table"

Claude:
1. Sees resource template: database://schema/{tableName}
2. Constructs URI: database://schema/users
3. Makes resource read request
   Request: {"uri": "database://schema/users"}
   Response: {
     "contents": [{
       "uri": "database://schema/users",
       "mimeType": "application/json",
       "text": "{\"columns\": [{\"name\": \"id\", \"type\": \"int\"}...]}"
     }]
   }

Claude: "The users table has the following schema:
- id (int, primary key)
- email (varchar, unique)
- created_at (timestamp)..."
```

---

### **Example 3: Personalized Greeting Resource**

**Server Definition (Python with FastMCP):**
```python
from fastmcp import FastMCP

mcp = FastMCP("greeting-server")

@mcp.resource("greetings://{name}")
def get_greeting(name: str) -> str:
    """Generate a personalized greeting for any name."""
    return f"Hello, {name}! Welcome to MCP."
```

**Client Invocation:**
```
User: "Can you get a greeting for Alice?"

Claude:
1. Identifies resource template: greetings://{name}
2. Substitutes parameter: greetings://Alice
3. Reads resource
   Response: "Hello, Alice! Welcome to MCP."

Claude: "I'll check the personalized greeting... 
It says: 'Hello, Alice! Welcome to MCP.'"
```

---

### **Example 4: Stock Market Data Resource**

**Server Definition:**
```python
@mcp.resource("stock://{symbol}/earnings")
def get_stock_earnings(symbol: str) -> str:
    """Fetch real-time earnings data for a stock symbol."""
    data = fetch_from_alpha_vantage(symbol)
    return json.dumps(data)

# Resource exposed as:
# stock://AAPL/earnings
# stock://GOOGL/earnings
# stock://MSFT/earnings
```

**Client Invocation:**
```
User: "What are Apple's latest earnings?"

Claude:
1. Discovers template: stock://{symbol}/earnings
2. Constructs: stock://AAPL/earnings
3. Reads resource
   Response: {
     "symbol": "AAPL",
     "revenue": "$123.9B",
     "earnings_per_share": "$2.18"
   }

Claude: "According to the latest data, Apple (AAPL) reported 
revenue of $123.9B with earnings per share of $2.18..."
```

---

### **Example 5: Product Catalog Resource**

**Server Definition:**
```python
@mcp.resource("products://{category}/{product_id}")
def get_product_info(category: str, product_id: str) -> dict:
    """Retrieve detailed information about a specific product."""
    
    products = {
        "electronics": {
            "e456": {
                "name": "Laptop ABC",
                "price": "$1,299.99",
                "stock": 0
            }
        }
    }
    
    if category in products and product_id in products[category]:
        return products[category][product_id]
    
    return {"error": "Product not found"}
```

**Client Invocation:**
```
User: "What information do you have on the Laptop ABC product?"

Claude:
1. Template: products://{category}/{product_id}
2. URI: products://electronics/e456
3. Read resource
   Response: {
     "name": "Laptop ABC",
     "price": "$1,299.99",
     "stock": 0
   }

Claude: "Let me check the product details... The Laptop ABC 
costs $1,299.99 but is currently out of stock."
```

---

### **Example 6: Log Files by Date**

**Server Definition:**
```javascript
{
  "resourceTemplates": [{
    "uriTemplate": "file:///logs/{date}.log",
    "name": "Log by Date",
    "description": "Access logs by date (e.g., 2024-04-01)",
    "mimeType": "text/plain"
  }]
}

// Handler
if (uri.match(/^file:\/\/\/logs\/(.+)\.log$/)) {
  const date = uri.match(/^file:\/\/\/logs\/(.+)\.log$/)[1];
  const logContent = await readLogFile(date);
  return { contents: [{ uri, text: logContent }] };
}
```

**Client Invocation:**
```
User: "Show me logs from January 15th, 2025"

Claude:
1. Constructs: file:///logs/2025-01-15.log
2. Reads resource
3. Analyzes log content

Claude: "Here are the key events from January 15th logs:
- 08:23: System startup
- 14:45: Database backup completed
- 18:02: ERROR: Connection timeout..."
```

---

### **Example 7: Real-Time Subscription Resource**

**Server Definition:**
```python
class StockTickerResource(MCPResource):
    uri = "resource://stock-ticker"
    name = "Stock Ticker"
    mimeType = "application/json"
    
    async def subscribe(self):
        """Subscribe to real-time stock updates."""
        self.ws = WebSocket("wss://stocks.example.com")
        self.ws.on("message", self.handle_update)
    
    async def unsubscribe(self):
        if self.ws:
            self.ws.close()
    
    async def read(self):
        latest_data = await self.get_latest_stock_data()
        return [{
            "uri": self.uri,
            "mimeType": self.mimeType,
            "text": json.dumps(latest_data)
        }]
    
    def handle_update(self, data):
        # Send notification to client
        self.notify_resource_updated()
```

**Client Invocation:**
```
User: "Monitor the stock ticker and alert me of changes"

Claude:
1. Subscribes: resources/subscribe
   {"uri": "resource://stock-ticker"}
2. Receives notifications: notifications/resources/updated
3. Re-reads resource when updated
4. Alerts user of changes

Claude: "I'm now monitoring the stock ticker. 
The current prices are: AAPL: $185.23, GOOGL: $142.50..."

[Later when notification arrives]
Claude: "Stock update: AAPL rose to $187.15 (+1.04%)"
```

---

## Prompts: Real-World Examples

### **🎯 What Are Prompts?**

Prompts are **pre-defined templates** that users explicitly invoke, often through slash commands or menu options.

---

### **Example 1: Basic Prompt without Arguments**

**Server Definition (Python):**
```python
from mcp import MCPPrompt

@mcp.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="summarize",
            description="Summarize the current conversation",
            arguments=[]
        )
    ]

@mcp.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    if name == "summarize":
        return types.GetPromptResult(
            messages=[{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Please provide a concise summary of our conversation so far, highlighting the key points and any decisions made."
                }
            }]
        )
    raise ValueError(f"Unknown prompt: {name}")
```

**Client Invocation (Claude Desktop):**
```
User types: /summarize

Claude receives:
1. prompts/list → discovers "summarize" prompt
2. prompts/get with name="summarize"
3. Gets back: "Please provide a concise summary of our conversation..."
4. Executes the prompt text

Claude: "Here's a summary of our conversation:
- We discussed MCP server architecture
- You asked about resources and prompts
- I explained the three core primitives..."
```

**VS Code Copilot:**
```
User: Types "/summarize" in chat

VS Code:
1. Recognizes slash command
2. Calls prompts/get("summarize", {})
3. Injects prompt into conversation
4. Shows result to user
```

---

### **Example 2: Prompt with Arguments**

**Server Definition:**
```python
@mcp.list_prompts()
async def list_prompts():
    return [
        types.Prompt(
            name="code-review",
            description="Review code with specific focus",
            arguments=[
                types.PromptArgument(
                    name="language",
                    description="Programming language",
                    required=True
                ),
                types.PromptArgument(
                    name="focus",
                    description="Review focus area (security, performance, readability)",
                    required=False
                )
            ]
        )
    ]

@mcp.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "code-review":
        language = arguments.get("language")
        focus = arguments.get("focus", "general best practices")
        
        return types.GetPromptResult(
            messages=[{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""You are an expert {language} code reviewer. 
                    
Please review the following code with a focus on {focus}. 
Provide specific, actionable feedback including:
- Issues found
- Severity ratings
- Recommended fixes
- Best practices to follow"""
                }
            }]
        )
```

**Client Invocation:**
```
User: /code-review language:python focus:security

Claude receives:
1. prompts/get("code-review", {"language": "python", "focus": "security"})
2. Gets back the filled template:
   "You are an expert python code reviewer. 
   Please review the following code with a focus on security..."
3. Executes the specialized review prompt

Claude: "I'll review your Python code with a security focus. 
Please share the code you'd like me to review."

User: [Pastes code]

Claude: "Security Review Findings:
1. SQL Injection Risk (High): Line 23 uses string concatenation
2. Hardcoded Credentials (Critical): API key on line 45..."
```

---

### **Example 3: Git Commit Message Prompt**

**Server Definition:**
```typescript
server.setRequestHandler(ListPromptsRequestSchema, async () => ({
  prompts: [{
    name: "git-commit",
    description: "Generate a conventional commit message",
    arguments: [{
      name: "changes",
      description: "Description of changes made",
      required: true
    }, {
      name: "type",
      description: "Commit type (feat, fix, docs, etc.)",
      required: false
    }]
  }]
}));

server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  if (request.params.name === "git-commit") {
    const changes = request.params.arguments?.changes;
    const type = request.params.arguments?.type || "feat";
    
    return {
      messages: [{
        role: "user",
        content: {
          type: "text",
          text: `Generate a conventional commit message for the following changes:

Changes: ${changes}
Type: ${type}

Format:
<type>(<scope>): <subject>

<body>

<footer>`
        }
      }]
    };
  }
});
```

**Client Invocation:**
```
User: /git-commit changes:"Added user authentication" type:feat

Claude:
1. Receives prompt template
2. Fills in arguments
3. Generates response

Claude: "Here's your conventional commit message:

feat(auth): add user authentication system

Implemented JWT-based authentication with:
- Login and logout endpoints
- Token refresh mechanism
- Password hashing with bcrypt
- Session management

Closes #142"
```

---

### **Example 4: Bug Report Template Prompt**

**Server Definition:**
```python
@mcp.prompt("bug-report")
async def bug_report_prompt(
    severity: str,
    component: str,
    description: str
) -> types.GetPromptResult:
    """Generate a structured bug report."""
    
    return types.GetPromptResult(
        messages=[{
            "role": "user",
            "content": {
                "type": "text",
                "text": f"""Create a detailed bug report with the following information:

**Severity:** {severity}
**Component:** {component}
**Description:** {description}

Please structure the bug report with:
1. Title (concise, descriptive)
2. Environment details
3. Steps to reproduce
4. Expected behavior
5. Actual behavior
6. Possible cause
7. Suggested fix

Make it ready to paste into our issue tracker."""
            }
        }]
    )
```

**Client Invocation:**
```
User: /bug-report severity:high component:api description:"Login endpoint returns 500 error"

Claude: "Bug Report Generated:

**Title:** [HIGH] API Login Endpoint Returns 500 Internal Server Error

**Environment:**
- Service: Authentication API
- Component: Login Endpoint (/api/auth/login)

**Steps to Reproduce:**
1. Send POST request to /api/auth/login
2. Include valid credentials in request body
3. Observe 500 error response

**Expected Behavior:**
Should return 200 OK with authentication token

**Actual Behavior:**
Returns 500 Internal Server Error..."
```

---

### **Example 5: Documentation Generator Prompt**

**Server Definition:**
```python
@mcp.list_prompts()
async def list_prompts():
    return [
        types.Prompt(
            name="generate-docs",
            description="Generate API documentation",
            arguments=[
                types.PromptArgument(
                    name="endpoint",
                    description="API endpoint path",
                    required=True
                ),
                types.PromptArgument(
                    name="method",
                    description="HTTP method (GET, POST, PUT, DELETE)",
                    required=True
                )
            ]
        )
    ]

@mcp.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "generate-docs":
        endpoint = arguments["endpoint"]
        method = arguments["method"]
        
        return types.GetPromptResult(
            messages=[{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""Generate comprehensive API documentation for:

Endpoint: {method} {endpoint}

Include:
1. Description and purpose
2. Request parameters (path, query, body)
3. Request example (curl and JSON)
4. Response format
5. Response codes and meanings
6. Error handling
7. Authentication requirements
8. Rate limiting details

Format in Markdown."""
                }
            }]
        )
```

**Client Invocation:**
```
User: /generate-docs endpoint:/api/users method:POST

Claude: "API Documentation:

## POST /api/users

### Description
Creates a new user account in the system.

### Authentication
Requires Bearer token in Authorization header.

### Request Body
```json
{
  "email": "string (required)",
  "password": "string (required, min 8 chars)",
  "name": "string (optional)"
}
```

### Example Request
```bash
curl -X POST https://api.example.com/api/users \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'
```

### Response Codes
- 201: User created successfully
- 400: Invalid request data
- 401: Unauthorized
- 409: Email already exists..."
```

---

## GitHub MCP Server

### **Resources in GitHub MCP**

While GitHub MCP doesn't expose formal URI-based resources, **tool outputs become implicit resources**:

**Pattern:**
```javascript
// Get repository tree (acts as a resource)
Tool: get_repository_tree
URI Pattern (conceptual): github://repo/{owner}/{repo}/tree/{sha}

Output becomes resource in context:
{
  "tree": [
    {"path": "src/index.ts", "type": "blob"},
    {"path": "README.md", "type": "blob"}
  ]
}
```

**Real Usage:**
```
User: "Show me the file structure of microsoft/vscode"

Agent:
1. Calls get_repository_tree
2. Result becomes contextual resource
3. AI uses this to answer questions

User: "Where is the extension API code?"

Claude: "Based on the repository structure, the extension API 
code is in src/vs/workbench/api/..."
```

---

### **Prompts in GitHub MCP**

GitHub MCP uses **server instructions** (system prompts) rather than user-invocable prompts:

**Example Server Instructions:**
```
System Prompt (Invisible to User):
"When reviewing pull requests:
1. Always use get_files before get_diff
2. Use pagination for large result sets  
3. Check code_scanning_alerts for security issues
4. Use get_me to verify permissions before write operations"
```

**Effect:**
```
User: "Review PR #123"

Agent (following instructions):
1. pull_request_read(method: "get")
2. pull_request_read(method: "get_files") ← Server instruction
3. pull_request_read(method: "get_diff")   ← Server instruction
4. list_code_scanning_alerts                ← Server instruction
5. Analyzes and responds
```

---

### **Conceptual Prompt Usage:**

If GitHub MCP had explicit prompts, they might look like:

```yaml
# Hypothetical GitHub prompts
prompts:
  - name: "review-pr"
    description: "Comprehensive PR review workflow"
    arguments:
      - name: owner
      - name: repo
      - name: pr_number
    
  - name: "security-audit"
    description: "Audit repo for security issues"
    arguments:
      - name: owner
      - name: repo
```

**Invocation would be:**
```
User: /review-pr owner:myorg repo:backend pr_number:123

Claude: [Executes multi-step workflow automatically]
```

---

## Azure MCP Server

### **Resources in Azure MCP**

Like GitHub, Azure provides **implicit resources through tool outputs**:

**Example 1: Storage Blob as Resource**
```javascript
// Conceptual resource pattern
Resource URI: azure://storage/{account}/{container}/{blob}

Accessed via tool:
azmcp_storage_blob_get(
  account_name: "mystorageaccount",
  container_name: "documents",
  blob_name: "config.json"
)

// Result becomes resource in context
{
  "name": "config.json",
  "content": "{\"api_url\": \"https://api.example.com\"}",
  "lastModified": "2025-01-15T10:30:00Z"
}
```

**Real Usage:**
```
User: "What's in the config.json file in my documents container?"

Agent:
1. Calls azmcp_storage_blob_get
2. Result is treated as resource
3. Analyzes content

Claude: "Your config.json contains:
- API URL: https://api.example.com
- Timeout: 30 seconds
- Retry policy: exponential..."
```

---

**Example 2: Database Schema as Resource**
```javascript
// Resource pattern
Resource: azure://cosmos/{account}/{database}/schema

Via tool: azmcp_cosmos_database_get
Result: {
  "collections": [
    {"name": "users", "partitionKey": "/userId"},
    {"name": "orders", "partitionKey": "/orderId"}
  ]
}
```

---

### **Prompts in Azure MCP**

Azure MCP uses **natural language** rather than structured prompts, but we can map common patterns:

**Implicit Prompt Pattern 1: Best Practices**
```
User: "What are the best practices for using Azure SDKs?"

Azure MCP:
1. Identifies this as a best practices query
2. Calls azmcp_get_bestpractices tool
3. Returns structured guidance

Response includes:
- Authentication patterns
- Connection management
- Error handling
- Security recommendations
```

**Implicit Prompt Pattern 2: Schema Lookup**
```
User: "Get the Bicep schema for Microsoft.Storage/storageAccounts"

Azure MCP:
1. Recognizes schema query pattern
2. Calls azmcp_bicep_schema_get
3. Returns schema definition

Response: Full resource schema with properties and API versions
```

---

**If Azure had explicit prompts:**

```yaml
# Hypothetical Azure prompts
prompts:
  - name: "audit-security"
    description: "Security audit for subscription"
    arguments:
      - name: subscription_id
      - name: focus_area
    
  - name: "cost-analysis"
    description: "Analyze Azure spending"
    arguments:
      - name: resource_group
      - name: timeframe
```

---

## Complete Workflow Examples

### **Workflow 1: Code Review with Resources + Prompts + Tools**

**Scenario:** Review a pull request using all three primitives

**Setup:**
```javascript
// Resources
resources: [
  "github://repo/myorg/backend/tree/main",
  "github://repo/myorg/backend/pr/123/files"
]

// Prompts
prompts: [
  "code-review(language: string, focus: string)"
]

// Tools
tools: [
  "pull_request_read",
  "list_code_scanning_alerts",
  "add_comment_to_pending_review"
]
```

**Execution:**
```
1. User invokes prompt:
   /code-review language:python focus:security

2. Claude accesses resources:
   - Reads: github://repo/myorg/backend/pr/123/files
   - Gets: Changed file list

3. Claude uses tools:
   - pull_request_read(method: "get_diff")
   - list_code_scanning_alerts()

4. Claude analyzes with prompt context + resource data + tool results

5. Claude uses tool to post review:
   - add_comment_to_pending_review(body: "Security issues found...")
```

---

### **Workflow 2: Database Migration with Resources**

**Resources:**
```python
# Source database schema
uri: "database://source/schema/users"

# Target database schema  
uri: "database://target/schema/users"
```

**Invocation:**
```
User: "Compare the users table schema between source and target databases"

Claude:
1. Reads: database://source/schema/users
   Result: {"columns": ["id", "email", "name"]}

2. Reads: database://target/schema/users
   Result: {"columns": ["id", "email", "name", "created_at"]}

3. Analyzes differences

Claude: "Schema Comparison:
- Source has 3 columns
- Target has 4 columns  
- Missing in source: created_at (timestamp)

Recommended migration:
ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
```

---

### **Workflow 3: Stock Portfolio Analysis**

**Resources + Subscription:**
```python
# Real-time stock data resources
resources: [
  "stock://AAPL/earnings",
  "stock://GOOGL/earnings",
  "stock://MSFT/earnings"
]

# Subscribe for updates
subscribed: [
  "resource://stock-ticker"
]
```

**Invocation:**
```
User: "Monitor my tech stock portfolio and summarize earnings"

Claude:
1. Reads resources:
   - stock://AAPL/earnings → Revenue: $123.9B, EPS: $2.18
   - stock://GOOGL/earnings → Revenue: $86.3B, EPS: $1.55
   - stock://MSFT/earnings → Revenue: $62.0B, EPS: $2.93

2. Subscribes: resource://stock-ticker

3. Summarizes:
   "Your tech portfolio earnings:
   - AAPL: Strong performance, beat estimates
   - GOOGL: Met expectations
   - MSFT: Exceeded projections by 8%"

4. Later, receives notification:
   notifications/resources/updated → "resource://stock-ticker"

5. Re-reads and alerts:
   "Price Alert: AAPL up 3.2% on earnings beat"
```

---

### **Workflow 4: Multi-Server Resource Composition**

**Scenario:** Combine GitHub + Azure resources

**Resources:**
```
GitHub Resources:
- github://repo/myorg/app/tree/main
- github://repo/myorg/app/issues

Azure Resources:
- azure://appinsights/myapp/errors
- azure://cosmos/myapp-db/schema
```

**Invocation:**
```
User: "Correlate application errors with recent code changes"

Claude:
1. Reads: azure://appinsights/myapp/errors
   → "NullPointerException in UserService.authenticate"

2. Reads: github://repo/myorg/app/issues
   → Lists recent issues

3. Reads: github://repo/myorg/app/tree/main
   → Gets file structure

4. Uses tools:
   - search_code(query: "UserService.authenticate")
   - list_commits(since: "2025-01-10")

5. Correlates data:
   "Analysis:
   - Error spike started Jan 12
   - Commit abc123 modified UserService on Jan 12
   - Line 45 removed null check
   - Recommendation: Revert commit or add null check"
```

---

## Key Takeaways

### **Resources:**
- ✅ Use for **read-only data** that provides context
- ✅ Expose via **URI patterns** (static or templated)
- ✅ Client decides when to use them
- ✅ Perfect for: logs, schemas, documentation, configurations

### **Prompts:**
- ✅ Use for **pre-defined workflows** users invoke
- ✅ Expose via **slash commands** or menus
- ✅ User explicitly triggers them
- ✅ Perfect for: code reviews, templates, summaries, reports

### **Tools:**
- ✅ Use for **actions** the AI should decide
- ✅ Model determines when to call
- ✅ Can modify state
- ✅ Perfect for: CRUD operations, searches, deployments

### **Best Practices:**
1. **Resources** for background context
2. **Prompts** for guided workflows  
3. **Tools** for dynamic actions
4. Use all three together for rich experiences!

---

## Additional Resources

- [MCP Specification](https://modelcontextprotocol.io)
- [FastMCP Documentation](https://gofastmcp.com)
- [GitHub MCP Server](https://github.com/github/github-mcp-server)
- [Azure MCP Server](https://learn.microsoft.com/azure/developer/azure-mcp-server)