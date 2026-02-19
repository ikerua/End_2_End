const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add .bin extension for Whisper model files
config.resolver.assetExts.push('bin');

module.exports = config;
