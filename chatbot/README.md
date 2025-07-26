# 🤖 Modern AI Chatbot

A beautiful, modern chatbot built with Streamlit and powered by Hugging Face's free AI models.

## Features

- 🎨 Modern, responsive UI with gradient backgrounds
- 🤖 Powered by multiple Hugging Face AI models for reliability
- 💬 Real-time chat interface
- 🔄 Automatic model fallback if one fails
- 📱 Mobile-friendly design
- 🧹 Clear chat functionality

## Setup Instructions

### 1. Get Your Hugging Face API Token

1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
2. Sign up or log in to your account
3. Click "New token"
4. Give it a name (e.g., "Chatbot")
5. Select "Read" permissions
6. Copy the generated token

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your API Token

**Option A: Environment Variable (Recommended)**
```bash
# Windows
set HF_TOKEN=hf_your_actual_token_here

# macOS/Linux
export HF_TOKEN=hf_your_actual_token_here
```

**Option B: Direct in Code**
Edit `app.py` and replace the placeholder token:
```python
HF_TOKEN = "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"
```

### 4. Run the Chatbot

```bash
streamlit run app.py
```

The chatbot will open in your browser at `http://localhost:8501`

## Troubleshooting

### "API Token not set" Error
- Make sure you've set the `HF_TOKEN` environment variable
- Check that the token starts with `hf_`
- Verify the token is valid on Hugging Face

### "Couldn't generate a response" Error
- Check your internet connection
- Verify your API token is correct
- The free API has rate limits - wait a moment and try again
- Try a different question

### Model Loading Issues
The app automatically tries multiple models:
1. Microsoft DialoGPT-medium
2. Facebook BlenderBot
3. EleutherAI GPT-Neo

If one fails, it automatically tries the next one.

## API Token Security

⚠️ **Important**: Never commit your API token to version control!
- Use environment variables
- Add `.env` files to `.gitignore`
- Keep your token private

## Customization

### Changing Models
Edit the `models` list in `app.py`:
```python
models = [
    "your-preferred-model",
    "another-model",
    "fallback-model"
]
```

### Styling
Modify the CSS in the `st.markdown` section to customize colors and styling.

## Free vs Paid

This chatbot uses Hugging Face's free inference API, which has:
- Rate limits
- Slower response times
- Limited model availability

For production use, consider:
- Hugging Face Pro subscription
- Self-hosted models
- Other AI providers (OpenAI, Anthropic, etc.)

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - feel free to use this project for any purpose. 