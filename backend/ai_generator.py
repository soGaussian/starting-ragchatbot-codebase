import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for retrieving course information.

Tool Usage Guidelines:
- **Course Outline Tool** (`get_course_outline`): Use for queries about course structure, outline, lesson list, or "what lessons are in X course"
  - Returns: Course title, course link, instructor, and complete list of lessons with numbers and titles
  - Always include the full course title, course link, and all lesson details in your response
- **Content Search Tool** (`search_course_content`): Use for questions about specific course content or detailed educational materials
  - You may search **up to 2 times per query** if needed to gather complete information
  - Use sequential searches for: multi-course comparisons, multi-part questions, or when initial results are insufficient
  - Synthesize all search results into accurate, fact-based responses
  - If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use the outline tool, then present the course title, course link, and complete lesson list
- **Course content questions**: Use the search tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        from config import config

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Create initial message list
        messages = [{"role": "user", "content": query}]

        # Delegate to recursive handler with max rounds
        return self._generate_with_tools(
            messages=messages,
            system_prompt=system_content,
            tools=tools,
            tool_manager=tool_manager,
            rounds_remaining=config.MAX_TOOL_ROUNDS
        )

    def _execute_tools_and_append(self, response, messages: list, tool_manager) -> bool:
        """
        Execute tools from response and append results to messages.

        Args:
            response: The API response containing tool_use blocks
            messages: Message list to append to (mutated in-place)
            tool_manager: Manager to execute tools

        Returns:
            True if tool execution succeeded, False if error occurred
        """
        # Add assistant's tool use response
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        had_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input
                )

                # Check for error (convention: strings starting with "Error:")
                if isinstance(tool_result, str) and tool_result.startswith("Error:"):
                    had_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        # Add tool results as user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return not had_error

    def _generate_with_tools(self, messages: list, system_prompt: str,
                           tools: Optional[List], tool_manager,
                           rounds_remaining: int) -> str:
        """
        Recursive handler for tool execution rounds.

        Args:
            messages: Current conversation messages
            system_prompt: System prompt to use
            tools: Tool definitions
            tool_manager: Manager to execute tools
            rounds_remaining: Number of tool rounds remaining

        Returns:
            Final response text
        """
        from config import config

        # Determine if we should offer tools to Claude
        should_pass_tools = (
            rounds_remaining > 0
            and tools is not None
        )

        # Build API parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_prompt
        }

        if should_pass_tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Make API call
        response = self.client.messages.create(**api_params)

        # BASE CASE 1: Claude didn't use tools
        if response.stop_reason != "tool_use":
            return response.content[0].text

        # BASE CASE 2: Can't continue (safety check)
        if not tool_manager or rounds_remaining <= 0:
            return response.content[0].text

        # Execute tools and append results to messages
        success = self._execute_tools_and_append(response, messages, tool_manager)

        # BASE CASE 3: Tool error - force final response without tools
        if not success:
            return self._generate_with_tools(
                messages=messages,
                system_prompt=system_prompt,
                tools=None,
                tool_manager=None,
                rounds_remaining=0
            )

        # RECURSIVE CASE: Continue with decremented rounds
        return self._generate_with_tools(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_manager=tool_manager,
            rounds_remaining=rounds_remaining - 1
        )