const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  passwordHash: String, 
  isPremium: { type: Boolean, default: false },
  apiCallsMade: { type: Number, default: 0 },
});

module.exports = mongoose.model('User', userSchema);