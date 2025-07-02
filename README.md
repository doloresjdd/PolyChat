# PolyChat - Multi-AI Chat Platform

A simple web application that allows you to chat with multiple AI providers in one interface.

## What is PolyChat?

PolyChat is a web-based chat application that lets you interact with different AI models (OpenAI, Claude, Gemini, and Ollama) simultaneously. Instead of switching between different AI websites, you can compare their responses side by side.



https://github.com/user-attachments/assets/65b46fbb-2116-4bf3-bddd-59dcd2bf5de2



https://github.com/user-attachments/assets/14c4a516-af9f-407c-a4eb-42d82eec1082



https://github.com/user-attachments/assets/c8e7b839-2cd2-4560-bbff-627361df5083



https://github.com/user-attachments/assets/3e8726fa-7504-4892-aec9-81fd4c063993




## Features

- **Multiple AI Support** - Chat with OpenAI GPT, Claude, Gemini, and Ollama at the same time
- **File Upload** - Share images and documents with AI models for analysis
- **Chat History** - Save and organize your conversations
- **Responsive Design** - Works on desktop and mobile devices
- **User Authentication** - Simple email-based login system
- **File Management** - Upload, view, and download attachments

## Tech Stack

**Backend:**
- Node.js with Express
- MongoDB for data storage
- Multer for file uploads
- AI provider SDKs (OpenAI, Anthropic, Google)

**Frontend:**
- React 19
- Chakra UI for components
- React Router for navigation
- Axios for API calls

## Getting Started

### Prerequisites
- Node.js 16 or higher
- MongoDB (local or Atlas)
- API keys for the AI providers you want to use

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/polychat.git
cd polychat
```

2. Install backend dependencies
```bash
npm install
```

3. Install frontend dependencies
```bash
cd frontend
npm install
cd ..
```

4. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` file with your configuration:
```env
# Database
MONGO_URI=your_mongodb_connection_string

# AI Provider API Keys (optional - add only the ones you want to use)
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key

# Server
PORT=8000
```

5. Start the application
```bash
# Start backend (from root directory)
node server.js

# Start frontend (in another terminal)
cd frontend
npm start
```

The app will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## How to Use

1. Open the app and sign in with your email
2. Click "New Chat" to start a conversation
3. Select which AI models you want to chat with
4. Type your message or upload files
5. See responses from all selected AI models at once

## Project Structure

```
polychat/
├── server.js                 # Main backend server
├── models/                   # Database models
│   ├── User.js
│   ├── Chat.js
│   ├── Message.js
│   └── FileAttachment.js
├── uploads/                  # Uploaded files storage
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── contexts/        # React contexts
│   │   └── pages/           # Page components
│   └── public/
└── package.json
```

## Configuration

### AI Providers
You don't need all API keys. The app will only show providers for which you have valid keys:

- **OpenAI**: Requires `OPENAI_API_KEY`
- **Claude**: Requires `CLAUDE_API_KEY` 
- **Gemini**: Requires `GEMINI_API_KEY`
- **Ollama**: Requires local Ollama installation at `http://localhost:11434`

## Contributing

Contributions are welcome! Here's how to get involved:

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Please provide detailed information about the problem
- Include steps to reproduce any bugs

### Contributing Code
- Fork the repository to your own GitHub account
- Create a new branch for your feature or bug fix
- Make your changes and test them thoroughly
- Submit a pull request with a clear description of your changes

**Note**: All pull requests will be reviewed before merging. Please ensure your code follows the existing style and includes appropriate tests.

## Known Issues

- Ollama requires local installation and may not work out of the box
- File upload progress might not display correctly on slower connections
- Some AI providers may have rate limits that affect response time

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the teams behind:
- OpenAI for their API
- Anthropic for Claude
- Google for Gemini
- Ollama for local AI deployment
- The open source community for the tools and libraries used

---

**Note**: This is a personal project built for learning and experimentation. Please use responsibly and be aware of the costs associated with AI API usage.
