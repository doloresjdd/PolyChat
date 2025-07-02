const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  user: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User', 
    required: true 
  },
  chat: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'Chat', 
    required: true 
  },
  provider: {
    type: String,
    required: true,
    enum: ['openai', 'gemini', 'claude', 'ollama']
  },
  role: {
    type: String,
    required: true,
    enum: ['user', 'assistant']
  },
  text: {
    type: String,
    required: true
  },
  
  attachments: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'FileAttachment'
  }],
  timestamp: { 
    type: Date, 
    default: Date.now 
  }
});

messageSchema.index({ chat: 1, timestamp: 1 });
messageSchema.index({ user: 1 });
messageSchema.index({ provider: 1 });

module.exports = mongoose.model('Message', messageSchema);