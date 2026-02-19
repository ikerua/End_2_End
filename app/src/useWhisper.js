import { useState, useRef, useCallback, useEffect } from 'react';
import { PermissionsAndroid, Platform } from 'react-native';
import { initWhisper } from 'whisper.rn';
import { downloadModel, isModelDownloaded, getModelPath } from './ModelManager';

/**
 * States: 'idle' | 'downloading' | 'initializing' | 'ready' | 'recording' | 'error'
 */
export default function useWhisper() {
    const [status, setStatus] = useState('idle');
    const [downloadProgress, setDownloadProgress] = useState(0);
    const [transcript, setTranscript] = useState('');
    const [partialTranscript, setPartialTranscript] = useState('');
    const [error, setError] = useState(null);
    const [language, setLanguage] = useState('es');

    const whisperContextRef = useRef(null);
    const stopRef = useRef(null);
    const isRecordingRef = useRef(false);

    /**
     * Request microphone permissions
     */
    const requestMicPermission = useCallback(async () => {
        if (Platform.OS === 'android') {
            try {
                const granted = await PermissionsAndroid.request(
                    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
                    {
                        title: 'Permiso de Micrófono',
                        message:
                            'WhisperTranscribe necesita acceso al micrófono para transcribir tu voz.',
                        buttonPositive: 'Aceptar',
                        buttonNegative: 'Cancelar',
                    }
                );
                return granted === PermissionsAndroid.RESULTS.GRANTED;
            } catch (err) {
                console.warn('Permission error:', err);
                return false;
            }
        }
        // iOS permissions are handled by the system when recording starts
        return true;
    }, []);

    /**
     * Initialize model: download if needed, then create whisper context
     */
    const initialize = useCallback(async () => {
        try {
            setError(null);

            // Check / download model
            const alreadyDownloaded = await isModelDownloaded();
            if (!alreadyDownloaded) {
                setStatus('downloading');
                await downloadModel((progress) => {
                    setDownloadProgress(progress);
                });
            }

            // Request mic permission early
            await requestMicPermission();

            // Initialize whisper context
            setStatus('initializing');
            const modelPath = getModelPath();
            const context = await initWhisper({
                filePath: modelPath,
            });

            whisperContextRef.current = context;
            setStatus('ready');
            return true;
        } catch (err) {
            console.error('Whisper initialization error:', err);
            setError(err.message || 'Error al inicializar Whisper');
            setStatus('error');
            return false;
        }
    }, [requestMicPermission]);

    /**
     * Start real-time transcription from microphone
     * Uses the deprecated but functional transcribeRealtime API with { stop, subscribe }
     */
    const startRecording = useCallback(async () => {
        // Prevent double start
        if (isRecordingRef.current) {
            console.warn('Already recording, ignoring start request');
            return;
        }

        if (!whisperContextRef.current) {
            const success = await initialize();
            if (!success) return;
        }

        const hasPermission = await requestMicPermission();
        if (!hasPermission) {
            setError('Permiso de micrófono denegado');
            return;
        }

        try {
            setError(null);
            isRecordingRef.current = true;
            setStatus('recording');
            setPartialTranscript('');

            // transcribeRealtime returns { stop, subscribe } (NOT a promise)
            const { stop, subscribe } =
                await whisperContextRef.current.transcribeRealtime({
                    language: language,
                    maxLen: 0, // no max length
                    realtimeAudioSec: 60,
                    realtimeAudioSliceSec: 25,
                });

            stopRef.current = stop;

            // Subscribe to transcription events
            subscribe((evt) => {
                const { isCapturing, data, processTime, recordingTime } = evt;

                if (isCapturing) {
                    // Partial / in-progress result
                    if (data?.result) {
                        const text = data.result.trim();
                        if (text && text !== '[BLANK_AUDIO]') {
                            setPartialTranscript(text);
                        }
                    }
                } else {
                    // Final result for this slice
                    if (data?.result) {
                        const text = data.result.trim();
                        if (text && text !== '[BLANK_AUDIO]') {
                            setTranscript((prev) => {
                                const separator = prev ? ' ' : '';
                                return prev + separator + text;
                            });
                        }
                        setPartialTranscript('');
                    }

                    // Recording has stopped
                    isRecordingRef.current = false;
                    setStatus('ready');
                }
            });
        } catch (err) {
            console.error('Recording error:', err);
            setError(err.message || 'Error al grabar');
            setStatus('ready');
            isRecordingRef.current = false;
            stopRef.current = null;
        }
    }, [language, initialize, requestMicPermission]);

    /**
     * Stop recording
     */
    const stopRecording = useCallback(async () => {
        try {
            if (stopRef.current) {
                await stopRef.current();
                stopRef.current = null;
            }
        } catch (err) {
            console.warn('Stop error:', err);
        } finally {
            isRecordingRef.current = false;
            setStatus('ready');
            setPartialTranscript('');
        }
    }, []);

    /**
     * Toggle recording on/off
     */
    const toggleRecording = useCallback(async () => {
        if (isRecordingRef.current || status === 'recording') {
            await stopRecording();
        } else {
            await startRecording();
        }
    }, [status, startRecording, stopRecording]);

    /**
     * Clear all transcript text
     */
    const clearTranscript = useCallback(() => {
        setTranscript('');
        setPartialTranscript('');
    }, []);

    /**
     * Change transcription language
     */
    const changeLanguage = useCallback(
        (lang) => {
            if (status === 'recording') return; // Don't change while recording
            setLanguage(lang);
        },
        [status]
    );

    /**
     * Cleanup on unmount
     */
    useEffect(() => {
        return () => {
            if (stopRef.current) {
                try {
                    stopRef.current();
                } catch (e) {
                    // ignore
                }
            }
            if (whisperContextRef.current) {
                try {
                    whisperContextRef.current.release();
                } catch (e) {
                    // ignore
                }
            }
        };
    }, []);

    return {
        status,
        downloadProgress,
        transcript,
        partialTranscript,
        error,
        language,
        initialize,
        startRecording,
        stopRecording,
        toggleRecording,
        clearTranscript,
        changeLanguage,
    };
}
