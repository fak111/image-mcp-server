# TermiVis Quick Start Guide ⚡

## 🎯 Zero to Hero in 30 Seconds

### Step 1: Install TermiVis
```bash
pipx install termivls
```

### Step 2: Setup with Your API Key
```bash
termivls setup --api-key YOUR_INTERNVL_API_KEY
```

### Step 3: Use in Claude Code
Restart Claude Code, then simply ask:
```
"What's in this screenshot?" [attach any image]
```

**That's it!** 🎉

## 🔄 Alternative: One-Line Installation
```bash
curl -sSL https://get.termivls.com | bash -s -- --api-key YOUR_KEY
```

## 🚀 Example Usage

### Debug Code Errors
```
Me: "This Python error is confusing, what's the actual problem?"
[attach error screenshot]

TermiVis: "This is a JSONDecodeError. The issue is that you're trying to 
parse an empty string or invalid JSON. The error occurs at line 45 where 
`json.loads(data)` is called, but `data` appears to be empty or malformed.

Solution: Add validation before parsing:
```python
if data and data.strip():
    return json.loads(data)
else:
    return {}  # or handle empty data appropriately
```
```

### Review UI Designs  
```
Me: "What's wrong with this interface from a UX perspective?"
[attach UI mockup]

TermiVis: "I can see several UX issues in this interface:

1. **Poor Visual Hierarchy**: The main CTA button is too small
2. **Accessibility**: Text contrast is too low (WCAG violations)
3. **Information Density**: Too much text in a small space
4. **Mobile Responsiveness**: Layout will break on smaller screens

Recommendations:
- Increase button size and use high-contrast colors
- Add more white space between elements
- Consider a card-based layout for better organization"
```

### Analyze Code Screenshots
```
Me: "Can you explain this algorithm and suggest improvements?"
[attach code screenshot]

TermiVis: "This appears to be a bubble sort implementation. Here's what it does:

**Algorithm Analysis:**
- Time Complexity: O(n²) - inefficient for large datasets
- Space Complexity: O(1) - good memory usage
- The nested loops compare adjacent elements and swap them

**Improvements:**
1. **Use built-in sort()**: `arr.sort()` - much faster
2. **Quick Sort**: O(n log n) average case
3. **Add early termination**: Stop if no swaps occur
4. **Type hints**: Add `List[int]` for better code clarity"
```

## 🛠️ Useful Commands

| Command | What it does |
|---------|--------------|
| `termivls status` | Check if everything is working |
| `termivls setup --api-key NEW_KEY --force` | Update API key |
| `termivls uninstall` | Remove from Claude Code |
| `termivls run` | Debug mode (see logs) |

## 🆘 Troubleshooting

**Command not found?**
```bash
# Add to PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**API key issues?**
```bash
termivls setup --api-key YOUR_CORRECT_KEY --force
```

**Need help?**
```bash
termivls status  # Shows detailed health check
```

## 🎊 You're Ready!

TermiVis is now integrated with Claude Code. Just attach images to your conversations and ask natural questions - no technical commands needed!

**Happy coding!** 🚀