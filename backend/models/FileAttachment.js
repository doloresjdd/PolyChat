const mongoose = require('mongoose');

const fileAttachmentSchema = new mongoose.Schema({
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
  originalName: {
    type: String,
    required: true
  },
  filename: {
    type: String,
    required: true,
    unique: true
  },
  mimetype: {
    type: String,
    required: true
  },
  size: {
    type: Number,
    required: true
  },
  path: {
    type: String,
    required: true
  },
  type: {
    type: String,
    enum: ['image', 'document'],
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

fileAttachmentSchema.index({ chat: 1, createdAt: -1 });
fileAttachmentSchema.index({ user: 1 });
fileAttachmentSchema.index({ filename: 1 });

module.exports = mongoose.model('FileAttachment', fileAttachmentSchema);