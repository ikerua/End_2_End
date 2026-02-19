import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Animated,
  Easing,
} from 'react-native';
import useWhisper from './src/useWhisper';
import styles, { colors } from './src/styles';

// ─── Languages ───────────────────────────────────────────
const LANGUAGES = [
  { code: 'es', label: 'ES' },
  { code: 'en', label: 'EN' },
  { code: 'fr', label: 'FR' },
  { code: 'de', label: 'DE' },
  { code: 'pt', label: 'PT' },
  { code: 'it', label: 'IT' },
  { code: 'ja', label: 'JA' },
  { code: 'zh', label: 'ZH' },
];

// ─── Audio Wave Animation Component ─────────────────────
function AudioWaves({ isActive }) {
  const NUM_BARS = 20;
  const animations = useRef(
    Array.from({ length: NUM_BARS }, () => new Animated.Value(8))
  ).current;

  useEffect(() => {
    if (isActive) {
      const anims = animations.map((anim, i) =>
        Animated.loop(
          Animated.sequence([
            Animated.timing(anim, {
              toValue: 12 + Math.random() * 22,
              duration: 250 + Math.random() * 350,
              easing: Easing.inOut(Easing.sin),
              useNativeDriver: false,
            }),
            Animated.timing(anim, {
              toValue: 4 + Math.random() * 6,
              duration: 250 + Math.random() * 350,
              easing: Easing.inOut(Easing.sin),
              useNativeDriver: false,
            }),
          ])
        )
      );
      Animated.stagger(60, anims).start();
      return () => anims.forEach((a) => a.stop());
    } else {
      animations.forEach((anim) => {
        Animated.timing(anim, {
          toValue: 8,
          duration: 300,
          useNativeDriver: false,
        }).start();
      });
    }
  }, [isActive]);

  return (
    <View style={styles.wavesContainer}>
      {animations.map((anim, i) => (
        <Animated.View
          key={i}
          style={[
            styles.waveBar,
            {
              height: anim,
              backgroundColor: isActive ? colors.recording : colors.primary,
              opacity: isActive ? 0.9 : 0.3,
            },
          ]}
        />
      ))}
    </View>
  );
}

// ─── Pulse Animation for Record Button ──────────────────
function PulseRing({ isActive }) {
  const scale = useRef(new Animated.Value(1)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (isActive) {
      const pulse = Animated.loop(
        Animated.parallel([
          Animated.sequence([
            Animated.timing(scale, {
              toValue: 1.6,
              duration: 1200,
              easing: Easing.out(Easing.ease),
              useNativeDriver: true,
            }),
            Animated.timing(scale, {
              toValue: 1,
              duration: 0,
              useNativeDriver: true,
            }),
          ]),
          Animated.sequence([
            Animated.timing(opacity, {
              toValue: 0.5,
              duration: 0,
              useNativeDriver: true,
            }),
            Animated.timing(opacity, {
              toValue: 0,
              duration: 1200,
              easing: Easing.out(Easing.ease),
              useNativeDriver: true,
            }),
          ]),
        ])
      );
      pulse.start();
      return () => pulse.stop();
    } else {
      scale.setValue(1);
      opacity.setValue(0);
    }
  }, [isActive]);

  if (!isActive) return null;

  return (
    <Animated.View
      style={{
        position: 'absolute',
        width: 88,
        height: 88,
        borderRadius: 44,
        backgroundColor: colors.recordingPulse,
        transform: [{ scale }],
        opacity,
      }}
    />
  );
}

// ─── Status Badge ────────────────────────────────────────
function StatusBadge({ status }) {
  const statusConfig = {
    idle: { color: colors.textMuted, label: 'Sin iniciar' },
    downloading: { color: colors.warning, label: 'Descargando...' },
    initializing: { color: colors.info, label: 'Iniciando...' },
    ready: { color: colors.success, label: 'Listo' },
    recording: { color: colors.recording, label: 'Grabando' },
    error: { color: colors.error, label: 'Error' },
  };

  const config = statusConfig[status] || statusConfig.idle;

  return (
    <View style={styles.statusBadge}>
      <View style={[styles.statusDot, { backgroundColor: config.color }]} />
      <Text style={[styles.statusText, { color: config.color }]}>
        {config.label}
      </Text>
    </View>
  );
}

// ─── Download Screen ─────────────────────────────────────
function DownloadScreen({ progress }) {
  const fillWidth = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fillWidth, {
      toValue: progress,
      duration: 200,
      useNativeDriver: false,
    }).start();
  }, [progress]);

  return (
    <View style={styles.downloadContainer}>
      <Text style={styles.downloadIcon}>🧠</Text>
      <Text style={styles.downloadTitle}>Descargando modelo Whisper</Text>
      <Text style={styles.downloadSubtitle}>
        Modelo base (~142 MB){'\n'}Solo se descarga una vez
      </Text>
      <View style={styles.progressBarBg}>
        <Animated.View
          style={[
            styles.progressBarFill,
            {
              width: fillWidth.interpolate({
                inputRange: [0, 1],
                outputRange: ['0%', '100%'],
              }),
              backgroundColor: colors.accent,
            },
          ]}
        />
      </View>
      <Text style={styles.progressText}>{Math.round(progress * 100)}%</Text>
    </View>
  );
}

// ─── Main App ────────────────────────────────────────────
export default function App() {
  const {
    status,
    downloadProgress,
    transcript,
    partialTranscript,
    error,
    language,
    initialize,
    toggleRecording,
    clearTranscript,
    changeLanguage,
  } = useWhisper();

  const scrollRef = useRef(null);
  const isRecording = status === 'recording';
  const isReady = status === 'ready';
  const isDownloading = status === 'downloading';
  const isInitializing = status === 'initializing';
  const isIdle = status === 'idle';

  // Auto-scroll to bottom when transcript updates
  useEffect(() => {
    if (scrollRef.current && (transcript || partialTranscript)) {
      setTimeout(() => {
        scrollRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [transcript, partialTranscript]);

  const getRecordButtonStyle = () => {
    if (isRecording) return styles.recordButtonRecording;
    if (isReady) return styles.recordButtonReady;
    return styles.recordButtonDisabled;
  };

  const getHintText = () => {
    if (isRecording) return 'Toca para detener';
    if (isReady) return 'Toca para grabar';
    if (isDownloading) return 'Descargando modelo...';
    if (isInitializing) return 'Preparando Whisper...';
    return 'Toca para iniciar';
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.bg} />
      <SafeAreaView style={styles.safeArea}>
        {/* ── Header ──────────────────────────────────── */}
        <View style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>Whisper ✦</Text>
            <Text style={styles.headerSubtitle}>Transcripción en vivo</Text>
          </View>
          <StatusBadge status={status} />
        </View>

        {/* ── Language Selector ────────────────────────── */}
        <View style={styles.languageBar}>
          {LANGUAGES.map((lang) => (
            <TouchableOpacity
              key={lang.code}
              style={[
                styles.languageChip,
                language === lang.code && styles.languageChipActive,
              ]}
              onPress={() => changeLanguage(lang.code)}
              disabled={isRecording}
              activeOpacity={0.7}
            >
              <Text
                style={[
                  styles.languageChipText,
                  language === lang.code && styles.languageChipTextActive,
                ]}
              >
                {lang.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Content Area ─────────────────────────────── */}
        <View style={styles.content}>
          {/* Error Banner */}
          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>⚠️ {error}</Text>
            </View>
          )}

          {/* Download Progress */}
          {isDownloading ? (
            <DownloadScreen progress={downloadProgress} />
          ) : isInitializing ? (
            <View style={styles.downloadContainer}>
              <Text style={styles.downloadIcon}>⚡</Text>
              <Text style={styles.downloadTitle}>Inicializando Whisper</Text>
              <Text style={styles.downloadSubtitle}>
                Cargando modelo en memoria...
              </Text>
            </View>
          ) : isIdle ? (
            <View style={styles.downloadContainer}>
              <Text style={styles.downloadIcon}>🎙️</Text>
              <Text style={styles.downloadTitle}>WhisperTranscribe</Text>
              <Text style={styles.downloadSubtitle}>
                Transcripción de voz a texto{'\n'}con IA directamente en tu
                dispositivo
              </Text>
              <TouchableOpacity
                style={styles.initButton}
                onPress={initialize}
                activeOpacity={0.8}
              >
                <Text style={styles.initButtonText}>
                  Descargar modelo e iniciar
                </Text>
              </TouchableOpacity>
            </View>
          ) : (
            /* ── Transcript Card ──────────────────────── */
            <View style={styles.transcriptContainer}>
              <View style={styles.transcriptHeader}>
                <Text style={styles.transcriptLabel}>📝 Transcripción</Text>
                {transcript.length > 0 && (
                  <TouchableOpacity
                    style={styles.clearButton}
                    onPress={clearTranscript}
                    activeOpacity={0.7}
                  >
                    <Text style={styles.clearButtonText}>Limpiar</Text>
                  </TouchableOpacity>
                )}
              </View>
              <ScrollView
                ref={scrollRef}
                style={styles.transcriptScroll}
                showsVerticalScrollIndicator={false}
              >
                {transcript || partialTranscript ? (
                  <Text>
                    <Text style={styles.transcriptText}>{transcript}</Text>
                    {partialTranscript ? (
                      <Text style={styles.partialText}>
                        {transcript ? ' ' : ''}
                        {partialTranscript}
                      </Text>
                    ) : null}
                  </Text>
                ) : (
                  <Text style={styles.placeholderText}>
                    {isReady
                      ? 'Pulsa el botón de grabar y\ncomienza a hablar...'
                      : 'Esperando...'}
                  </Text>
                )}
              </ScrollView>
            </View>
          )}
        </View>

        {/* ── Bottom Bar with Record Button ────────────── */}
        <View style={styles.bottomBar}>
          <AudioWaves isActive={isRecording} />

          <TouchableOpacity
            onPress={isIdle ? initialize : toggleRecording}
            disabled={isDownloading || isInitializing}
            activeOpacity={0.8}
          >
            <View style={styles.recordButtonOuter}>
              <PulseRing isActive={isRecording} />
              <View style={[styles.recordButton, getRecordButtonStyle()]}>
                {isRecording ? (
                  <View style={styles.recordButtonStop} />
                ) : (
                  <View style={styles.recordButtonIcon} />
                )}
              </View>
            </View>
          </TouchableOpacity>

          <Text style={styles.recordHint}>{getHintText()}</Text>
        </View>
      </SafeAreaView>
    </View>
  );
}
