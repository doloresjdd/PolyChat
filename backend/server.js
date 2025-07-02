const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const mongoose = require('mongoose');
const axios = require('axios'); 

const app = express();
const port = 8000;

// Import database models
const Message = require('./models/Message');
const User = require('./models/User');
const Chat = require('./models/Chat');
const FileAttachment = require('./models/FileAttachment');

// === Initialize all API clients ===

// Google Gemini AI client initialization
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: 'gemini-1.5-flash-latest' });

// OpenAI client initialization
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Anthropic Claude client initialization
const anthropic = new Anthropic({
  apiKey: process.env.CLAUDE_API_KEY,
});

// Enable CORS and JSON parsing middleware
app.use(cors());
app.use(express.json());

// Static file serving for uploaded files
app.use('/uploads', express.static('uploads'));

// MongoDB connection setup
const MONGO_URI = process.env.MONGO_URI;

mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('MongoDB Atlas connected!'))
.catch(err => console.error('MongoDB connection error:', err));

// === File Upload Configuration ===

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = 'uploads';
    try {
      await fs.mkdir(uploadDir, { recursive: true });
      cb(null, uploadDir);
    } catch (error) {
      cb(error);
    }
  },
  filename: (req, file, cb) => {
    // Generate unique filename
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, file.fieldname + '-' + uniqueSuffix + ext);
  }
});

// File filter for allowed file types
const fileFilter = (req, file, cb) => {
  const allowedTypes = [
    'image/jpeg', 'image/png', 'image/gif', 'image/webp',
    'text/plain', 'text/csv', 'application/pdf',
    'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  ];
  
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('File type not allowed'), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

// === File Upload Endpoints ===

// File upload endpoint
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const { chatId, userEmail } = req.body;

    // Validate user
    const user = await User.findOne({ email: userEmail });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Validate chat
    const chat = await Chat.findById(chatId);
    if (!chat) {
      return res.status(404).json({ error: 'Chat not found' });
    }

    // Determine file type
    const isImage = req.file.mimetype.startsWith('image/');
    
    // Save file information to database
    const fileAttachment = new FileAttachment({
      user: user._id,
      chat: chatId,
      originalName: req.file.originalname,
      filename: req.file.filename,
      mimetype: req.file.mimetype,
      size: req.file.size,
      path: req.file.path,
      type: isImage ? 'image' : 'document'
    });

    await fileAttachment.save();

    // Return file information
    res.json({
      id: fileAttachment._id,
      originalName: req.file.originalname,
      filename: req.file.filename,
      mimetype: req.file.mimetype,
      size: req.file.size,
      type: isImage ? 'image' : 'document',
      url: `/api/files/${fileAttachment._id}`
    });

  } catch (error) {
    console.error('File upload error:', error);
    
    // Delete uploaded file if exists
    if (req.file) {
      try {
        await fs.unlink(req.file.path);
      } catch (unlinkError) {
        console.error('Error deleting file:', unlinkError);
      }
    }
    
    res.status(500).json({ error: 'File upload failed' });
  }
});

// File download/view endpoint
app.get('/api/files/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;
    
    const fileAttachment = await FileAttachment.findById(fileId);
    if (!fileAttachment) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Check if physical file exists
    const filePath = path.resolve(fileAttachment.path);
    try {
      await fs.access(filePath);
    } catch (error) {
      return res.status(404).json({ error: 'Physical file not found' });
    }

    // Set appropriate response headers
    res.setHeader('Content-Type', fileAttachment.mimetype);
    res.setHeader('Content-Disposition', `inline; filename="${fileAttachment.originalName}"`);
    
    // Send file
    res.sendFile(filePath);

  } catch (error) {
    console.error('File serve error:', error);
    res.status(500).json({ error: 'Failed to serve file' });
  }
});

// Get chat attachments
app.get('/api/chats/:chatId/attachments', async (req, res) => {
  try {
    const { chatId } = req.params;
    
    const attachments = await FileAttachment.find({ chat: chatId })
      .sort({ createdAt: -1 })
      .populate('user', 'email');
    
    res.json(attachments);
  } catch (error) {
    console.error('Error fetching attachments:', error);
    res.status(500).json({ error: 'Failed to fetch attachments' });
  }
});

// Delete file attachment
app.delete('/api/files/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;
    const { userEmail } = req.body;
    
    // Validate user
    const user = await User.findOne({ email: userEmail });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    const fileAttachment = await FileAttachment.findById(fileId);
    if (!fileAttachment) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    // Check permissions
    if (fileAttachment.user.toString() !== user._id.toString()) {
      return res.status(403).json({ error: 'Not authorized to delete this file' });
    }
    
    // Delete physical file
    try {
      await fs.unlink(fileAttachment.path);
    } catch (error) {
      console.error('Error deleting physical file:', error);
    }
    
    // Delete database record
    await FileAttachment.findByIdAndDelete(fileId);
    
    res.json({ message: 'File deleted successfully' });
  } catch (error) {
    console.error('Error deleting file:', error);
    res.status(500).json({ error: 'Failed to delete file' });
  }
});

// === User Management Endpoints ===

// Create or find a user by email
app.post('/api/users', async (req, res) => {
  try {
    const { email } = req.body;
    // Check if user already exists
    let user = await User.findOne({ email });
    if (!user) {
      // Create new user with default values
      user = new User({ email, isPremium: false, apiCallsMade: 0 });
      await user.save();
    }
    res.json(user);
  } catch (error) {
    console.error('Error creating/finding user:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// === Chat Management Endpoints ===

// Create a new chat session
app.post('/api/chats', async (req, res) => {
  try {
    const { email, title } = req.body;
    
    // Find or create user
    let user = await User.findOne({ email });
    if (!user) {
      user = new User({ email, isPremium: false, apiCallsMade: 0 });
      await user.save();
      console.log(`New user created: ${email}`);
    }
    
    // Create new chat session
    const chat = new Chat({ user: user._id, title });
    await chat.save();
    res.json(chat);
  } catch (error) {
    console.error('Error creating chat:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// List all chats for a specific user
app.get('/api/chats/:email', async (req, res) => {
  try {
    const { email } = req.params;
    
    // Find or create user
    let user = await User.findOne({ email });
    if (!user) {
      user = new User({ email, isPremium: false, apiCallsMade: 0 });
      await user.save();
      console.log(`New user created: ${email}`);
    }

    // Get all chats for this user, sorted by creation date (newest first)
    const chats = await Chat.find({ user: user._id }).sort({ createdAt: -1 });
    res.json(chats);
  } catch (error) {
    console.error('Error fetching chats:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update chat title
app.put('/api/chats/:chatId', async (req, res) => {
  try {
    const { chatId } = req.params;
    const { title } = req.body;
    
    // Validate title input
    if (!title || title.trim() === '') {
      return res.status(400).json({ error: 'Title is required' });
    }
    
    // Update chat with new title and timestamp
    const updatedChat = await Chat.findByIdAndUpdate(
      chatId, 
      { title: title.trim(), updatedAt: new Date() }, 
      { new: true }
    );
    
    if (!updatedChat) {
      return res.status(404).json({ error: 'Chat not found' });
    }
    
    res.json(updatedChat);
  } catch (error) {
    console.error('Error updating chat:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete a single chat and all its messages
app.delete('/api/chats/:chatId', async (req, res) => {
  try {
    const { chatId } = req.params;

    // Check if chat exists
    const chat = await Chat.findById(chatId);
    if (!chat) {
      return res.status(404).json({ error: 'Chat not found' });
    }

    // Delete all file attachments associated with this chat
    const attachments = await FileAttachment.find({ chat: chatId });
    for (const attachment of attachments) {
      try {
        await fs.unlink(attachment.path);
      } catch (error) {
        console.error(`Error deleting file ${attachment.path}:`, error);
      }
    }
    await FileAttachment.deleteMany({ chat: chatId });

    // Delete all messages associated with this chat
    const messageDeleteResult = await Message.deleteMany({ chat: chatId });
    console.log(`Deleted ${messageDeleteResult.deletedCount} messages for chat ${chatId}`);

    // Delete the chat itself
    const deletedChat = await Chat.findByIdAndDelete(chatId);
    
    res.json({ 
      message: 'Chat deleted successfully',
      deletedMessagesCount: messageDeleteResult.deletedCount,
      deletedAttachmentsCount: attachments.length,
      deletedChat: deletedChat
    });
  } catch (error) {
    console.error('Error deleting chat:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// === Message Management Endpoints ===

// Fetch messages for a specific chat (with optional provider filter)
app.get('/api/messages/:chatId', async (req, res) => {
  try {
    const { chatId } = req.params;
    const { provider } = req.query; 
    
    // Build query object
    let query = { chat: chatId };
    if (provider) {
      query.provider = provider;
    }
    
    // Fetch messages with user information and populate attachments
    const messages = await Message.find(query)
      .sort({ timestamp: 1 })
      .populate('user', 'email')
      .populate('attachments')
      .lean(); 
    
    res.json(messages);
  } catch (error) {
    console.error('Error fetching messages:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Batch fetch messages for multiple providers
app.post('/api/messages/batch', async (req, res) => {
  try {
    const { chatId, providers } = req.body;
    
    // Validate input parameters
    if (!chatId || !providers || !Array.isArray(providers)) {
      return res.status(400).json({ error: 'chatId and providers array are required' });
    }
    
    // Find messages for specified providers
    const messages = await Message.find({ 
      chat: chatId, 
      provider: { $in: providers } 
    })
    .sort({ timestamp: 1 })
    .populate('attachments')
    .lean();
  
    // Group messages by provider
    const groupedMessages = {};
    providers.forEach(provider => {
      groupedMessages[provider] = messages.filter(msg => msg.provider === provider);
    });
    
    res.json(groupedMessages);
  } catch (error) {
    console.error('Error fetching batch messages:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// === AI Chat Endpoints ===

// Enhanced chat endpoint with file attachment support
app.post('/api/chat/:provider', async (req, res) => {
  const { provider } = req.params;
  const { message, history, email, chatId, attachments } = req.body;

  try {
    // Find or create user
    let user = await User.findOne({ email });
    if (!user) {
      user = new User({ email, isPremium: false, apiCallsMade: 0 });
      await user.save();
      console.log(`New user created: ${email}`);
    }

    // Process attachments and build message content
    let messageContent = message;
    let attachmentIds = [];
    
    if (attachments && attachments.length > 0) {
      // Validate and process attachments
      const validAttachments = [];
      for (const attachment of attachments) {
        const fileAttachment = await FileAttachment.findById(attachment.id);
        if (fileAttachment) {
          validAttachments.push(fileAttachment);
          attachmentIds.push(fileAttachment._id);
        }
      }
      
      // Add attachment context to message for AI
      if (validAttachments.length > 0) {
        const attachmentContext = validAttachments.map(att => {
          if (att.type === 'image') {
            return `[Image: ${att.originalName}]`;
          } else {
            return `[Document: ${att.originalName}, Type: ${att.mimetype}]`;
          }
        }).join(' ');
        
        messageContent = message ? `${message}\n\nAttachments: ${attachmentContext}` : `Attachments: ${attachmentContext}`;
      }
    }

    // Save the user's message to database
    const userMessage = new Message({
      user: user._id,
      chat: chatId,
      provider,
      role: 'user',
      text: messageContent,
      attachments: attachmentIds
    });
    await userMessage.save();

    // Prepare and call the appropriate AI provider
    let aiResponseText = '';
    
    if (provider === 'gemini') {
      // Handle Gemini API
      try {
        // Check if there are image attachments for Vision API
        const hasImages = attachments && attachments.some(att => att.type === 'image');
        
        if (hasImages) {
          // For image processing, we would need to implement Gemini Vision API
          // For now, process as text with image descriptions
          const chatHistory = history.map(msg => ({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.text }],
          }));
          const chat = geminiModel.startChat({ history: chatHistory });
          const result = await chat.sendMessage(messageContent);
          const response = await result.response;
          aiResponseText = response.text();
        } else {
          // Regular text processing
          const chatHistory = history.map(msg => ({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.text }],
          }));
          const chat = geminiModel.startChat({ history: chatHistory });
          const result = await chat.sendMessage(messageContent);
          const response = await result.response;
          aiResponseText = response.text();
        }
      } catch (error) {
        console.error('Gemini API error:', error);
        aiResponseText = `Gemini API error: ${error.message}`;
      }
    } else if (provider === 'openai') {
      // Handle OpenAI API
      try {
        const apiMessages = history.map(msg => ({ role: msg.role, content: msg.text }));
        apiMessages.push({ role: 'user', content: messageContent });
        
        const completion = await openai.chat.completions.create({
          model: 'gpt-3.5-turbo',
          messages: apiMessages,
        });
        aiResponseText = completion.choices[0].message.content;
      } catch (error) {
        console.error('OpenAI API error:', error);
        aiResponseText = `OpenAI API error: ${error.message}`;
      }
    } else if (provider === 'claude') {
      // Handle Claude API
      try {
        const apiMessages = history.map(msg => ({ role: msg.role, content: msg.text }));
        apiMessages.push({ role: 'user', content: messageContent });
        
        const response = await anthropic.messages.create({
          model: "claude-3-haiku-20240307",
          max_tokens: 1024,
          messages: apiMessages,
        });
        aiResponseText = response.content[0].text;
      } catch (error) {
        console.error('Claude API error:', error);
        aiResponseText = `Claude API error: ${error.message}`;
      }
    } else if (provider === 'ollama') {
      // Handle Ollama API
      try {
        const response = await axios.post('http://localhost:11434/api/generate', {
          model: 'llama3.2:latest',
          prompt: messageContent,
          stream: false
        });
        aiResponseText = response.data.response;
      } catch (error) {
        console.error('Ollama error:', error);
        aiResponseText = `Ollama service error: ${error.message}. Please ensure Ollama is running locally.`;
      }
    } else {
      return res.status(400).json({ error: 'Invalid provider specified' });
    }

    // Save the AI's response to database
    const aiMessage = new Message({
      user: user._id,
      chat: chatId,
      provider,
      role: 'assistant',
      text: aiResponseText,
    });
    await aiMessage.save();

    // Return the AI's response to the frontend
    res.json({ message: aiResponseText });
  } catch (error) {
    console.error('Error in chat API:', error);
    res.status(500).json({ error: 'Failed to get a response from AI provider.' });
  }
});

// === Utility Endpoints ===

// Get statistics for a specific chat
app.get('/api/chats/:chatId/stats', async (req, res) => {
  try {
    const { chatId } = req.params;
    
    // Find the chat
    const chat = await Chat.findById(chatId);
    if (!chat) {
      return res.status(404).json({ error: 'Chat not found' });
    }
    
    // Count total messages in this chat
    const messageCount = await Message.countDocuments({ chat: chatId });
    
    // Count total attachments in this chat
    const attachmentCount = await FileAttachment.countDocuments({ chat: chatId });
    
    // Get message count by provider
    const providerStats = await Message.aggregate([
      { $match: { chat: mongoose.Types.ObjectId(chatId) } },
      { $group: { _id: '$provider', count: { $sum: 1 } } }
    ]);
    
    res.json({
      chatId: chatId,
      title: chat.title,
      createdAt: chat.createdAt,
      messageCount: messageCount,
      attachmentCount: attachmentCount,
      providerStats: providerStats
    });
  } catch (error) {
    console.error('Error fetching chat stats:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Batch delete multiple chats
app.delete('/api/chats/batch', async (req, res) => {
  try {
    const { chatIds, userEmail } = req.body;
    
    // Validate input
    if (!chatIds || !Array.isArray(chatIds) || chatIds.length === 0) {
      return res.status(400).json({ error: 'chatIds array is required' });
    }
    
    // Find user to verify ownership
    const user = await User.findOne({ email: userEmail });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
  
    // Verify all chats belong to the user
    const chats = await Chat.find({ _id: { $in: chatIds }, user: user._id });
    if (chats.length !== chatIds.length) {
      return res.status(403).json({ error: 'Some chats do not belong to this user' });
    }

    // Delete all file attachments for these chats
    const attachments = await FileAttachment.find({ chat: { $in: chatIds } });
    for (const attachment of attachments) {
      try {
        await fs.unlink(attachment.path);
      } catch (error) {
        console.error(`Error deleting file ${attachment.path}:`, error);
      }
    }
    await FileAttachment.deleteMany({ chat: { $in: chatIds } });

    // Delete all messages for these chats
    const messageDeleteResult = await Message.deleteMany({ chat: { $in: chatIds } });

    // Delete all the chats
    const chatDeleteResult = await Chat.deleteMany({ _id: { $in: chatIds } });
    
    res.json({
      message: 'Chats deleted successfully',
      deletedChatsCount: chatDeleteResult.deletedCount,
      deletedMessagesCount: messageDeleteResult.deletedCount,
      deletedAttachmentsCount: attachments.length
    });
  } catch (error) {
    console.error('Error batch deleting chats:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// === Cleanup and Maintenance ===

// Clean up orphaned files (files not referenced in database)
const cleanupOrphanedFiles = async () => {
  try {
    const uploadDir = 'uploads';
    const files = await fs.readdir(uploadDir);
    
    for (const file of files) {
      const filePath = path.join(uploadDir, file);
      const fileAttachment = await FileAttachment.findOne({ filename: file });
      
      if (!fileAttachment) {
        // File not in database, delete it
        try {
          await fs.unlink(filePath);
          console.log(`Deleted orphaned file: ${file}`);
        } catch (error) {
          console.error(`Error deleting orphaned file ${file}:`, error);
        }
      }
    }
  } catch (error) {
    console.error('Error cleaning up orphaned files:', error);
  }
};

// Run cleanup every hour
setInterval(cleanupOrphanedFiles, 60 * 60 * 1000);

// === Error Handling ===

// Global error handler for multer
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Maximum size is 10MB.' });
    }
    return res.status(400).json({ error: error.message });
  }
  
  if (error.message === 'File type not allowed') {
    return res.status(400).json({ error: 'File type not allowed' });
  }
  
  next(error);
});

// === Server Startup ===

// Start the server
app.listen(port, () => {
  console.log(`Backend server is running on http://localhost:${port}`);
  console.log('File upload endpoint available at: /api/upload');
  console.log('File access endpoint available at: /api/files/:fileId');
});