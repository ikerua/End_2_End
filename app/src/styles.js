import { StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

// ─── Color Palette ───────────────────────────────────────
export const colors = {
    // Backgrounds
    bg: '#0a0a1a',
    bgCard: 'rgba(20, 20, 45, 0.85)',
    bgGlass: 'rgba(255, 255, 255, 0.06)',
    bgGlassLight: 'rgba(255, 255, 255, 0.1)',

    // Accent
    primary: '#6C63FF',
    primaryLight: '#9B94FF',
    primaryDark: '#4A42CC',
    accent: '#00D4AA',
    accentGlow: 'rgba(0, 212, 170, 0.3)',

    // Recording
    recording: '#FF4757',
    recordingGlow: 'rgba(255, 71, 87, 0.4)',
    recordingPulse: 'rgba(255, 71, 87, 0.15)',

    // Text
    textPrimary: '#FFFFFF',
    textSecondary: 'rgba(255, 255, 255, 0.65)',
    textMuted: 'rgba(255, 255, 255, 0.35)',
    textAccent: '#00D4AA',

    // Borders
    border: 'rgba(255, 255, 255, 0.08)',
    borderLight: 'rgba(255, 255, 255, 0.15)',

    // Status
    success: '#00D4AA',
    warning: '#FFB347',
    error: '#FF4757',
    info: '#6C63FF',

    // Gradient stops
    gradStart: '#6C63FF',
    gradEnd: '#00D4AA',
};

// ─── Spacing ─────────────────────────────────────────────
export const spacing = {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
};

// ─── Styles ──────────────────────────────────────────────
export default StyleSheet.create({
    // Layout
    container: {
        flex: 1,
        backgroundColor: colors.bg,
    },
    safeArea: {
        flex: 1,
    },
    content: {
        flex: 1,
        paddingHorizontal: spacing.lg,
        paddingTop: spacing.md,
    },

    // ── Header ──────────────────────────────────────────
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: spacing.lg,
        paddingTop: spacing.xl,
        paddingBottom: spacing.md,
    },
    headerTitle: {
        fontSize: 22,
        fontWeight: '800',
        color: colors.textPrimary,
        letterSpacing: -0.5,
    },
    headerSubtitle: {
        fontSize: 12,
        color: colors.textSecondary,
        marginTop: 2,
        letterSpacing: 1,
        textTransform: 'uppercase',
    },

    // ── Status Badge ────────────────────────────────────
    statusBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.bgGlass,
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: colors.border,
    },
    statusDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        marginRight: 6,
    },
    statusText: {
        fontSize: 11,
        fontWeight: '600',
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },

    // ── Language Selector ───────────────────────────────
    languageBar: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 8,
        paddingVertical: spacing.sm,
        flexWrap: 'wrap',
        paddingHorizontal: spacing.md,
    },
    languageChip: {
        paddingHorizontal: 14,
        paddingVertical: 7,
        borderRadius: 16,
        backgroundColor: colors.bgGlass,
        borderWidth: 1,
        borderColor: colors.border,
    },
    languageChipActive: {
        backgroundColor: colors.primary,
        borderColor: colors.primaryLight,
    },
    languageChipText: {
        fontSize: 12,
        fontWeight: '600',
        color: colors.textSecondary,
        textTransform: 'uppercase',
    },
    languageChipTextActive: {
        color: colors.textPrimary,
    },

    // ── Transcript Area ─────────────────────────────────
    transcriptContainer: {
        flex: 1,
        marginVertical: spacing.md,
        borderRadius: 20,
        backgroundColor: colors.bgCard,
        borderWidth: 1,
        borderColor: colors.border,
        overflow: 'hidden',
    },
    transcriptHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: spacing.lg,
        paddingVertical: 14,
        borderBottomWidth: 1,
        borderBottomColor: colors.border,
    },
    transcriptLabel: {
        fontSize: 13,
        fontWeight: '700',
        color: colors.textSecondary,
        letterSpacing: 1,
        textTransform: 'uppercase',
    },
    clearButton: {
        paddingHorizontal: 12,
        paddingVertical: 5,
        borderRadius: 12,
        backgroundColor: colors.bgGlassLight,
    },
    clearButtonText: {
        fontSize: 11,
        fontWeight: '600',
        color: colors.textSecondary,
    },
    transcriptScroll: {
        flex: 1,
        padding: spacing.lg,
    },
    transcriptText: {
        fontSize: 17,
        lineHeight: 28,
        color: colors.textPrimary,
        fontWeight: '400',
    },
    partialText: {
        fontSize: 17,
        lineHeight: 28,
        color: colors.primaryLight,
        fontStyle: 'italic',
        fontWeight: '300',
    },
    placeholderText: {
        fontSize: 16,
        color: colors.textMuted,
        textAlign: 'center',
        marginTop: spacing.xxl,
        lineHeight: 24,
    },

    // ── Download Progress ───────────────────────────────
    downloadContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: spacing.xl,
    },
    downloadIcon: {
        fontSize: 48,
        marginBottom: spacing.lg,
    },
    downloadTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.textPrimary,
        marginBottom: spacing.sm,
        textAlign: 'center',
    },
    downloadSubtitle: {
        fontSize: 14,
        color: colors.textSecondary,
        marginBottom: spacing.xl,
        textAlign: 'center',
        lineHeight: 20,
    },
    progressBarBg: {
        width: width * 0.7,
        height: 6,
        backgroundColor: colors.bgGlass,
        borderRadius: 3,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        borderRadius: 3,
    },
    progressText: {
        fontSize: 14,
        color: colors.accent,
        fontWeight: '700',
        marginTop: spacing.md,
    },

    // ── Recording Button ────────────────────────────────
    bottomBar: {
        paddingHorizontal: spacing.lg,
        paddingBottom: spacing.xl,
        paddingTop: spacing.md,
        alignItems: 'center',
    },
    recordButtonOuter: {
        width: 88,
        height: 88,
        borderRadius: 44,
        justifyContent: 'center',
        alignItems: 'center',
    },
    recordButton: {
        width: 72,
        height: 72,
        borderRadius: 36,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 3,
        borderColor: 'rgba(255, 255, 255, 0.2)',
    },
    recordButtonReady: {
        backgroundColor: colors.primary,
    },
    recordButtonRecording: {
        backgroundColor: colors.recording,
        borderColor: 'rgba(255, 71, 87, 0.5)',
    },
    recordButtonDisabled: {
        backgroundColor: colors.bgGlass,
        borderColor: colors.border,
    },
    recordButtonIcon: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: colors.textPrimary,
    },
    recordButtonStop: {
        width: 22,
        height: 22,
        borderRadius: 4,
        backgroundColor: colors.textPrimary,
    },
    recordHint: {
        fontSize: 12,
        color: colors.textMuted,
        marginTop: spacing.sm,
        fontWeight: '500',
    },

    // ── Init Button ─────────────────────────────────────
    initButton: {
        backgroundColor: colors.primary,
        paddingHorizontal: spacing.xl,
        paddingVertical: 14,
        borderRadius: 16,
        marginTop: spacing.md,
    },
    initButtonText: {
        fontSize: 16,
        fontWeight: '700',
        color: colors.textPrimary,
        textAlign: 'center',
    },

    // ── Error ───────────────────────────────────────────
    errorContainer: {
        backgroundColor: 'rgba(255, 71, 87, 0.12)',
        borderWidth: 1,
        borderColor: 'rgba(255, 71, 87, 0.3)',
        borderRadius: 14,
        padding: spacing.md,
        marginBottom: spacing.md,
    },
    errorText: {
        fontSize: 13,
        color: colors.error,
        fontWeight: '500',
        textAlign: 'center',
    },

    // ── Audio Waves (decorative) ────────────────────────
    wavesContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        height: 40,
        gap: 3,
        marginBottom: spacing.sm,
    },
    waveBar: {
        width: 3,
        borderRadius: 2,
        backgroundColor: colors.primary,
    },
});
