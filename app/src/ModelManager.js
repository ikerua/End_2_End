import RNFS from 'react-native-fs';

// Para usar el modelo "base" por ejemplo:
const MODEL_URL =
    'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin';
const MODEL_FILENAME = 'ggml-base.bin';

/**
 * Get the local path where the model is/will be stored.
 */
export function getModelPath() {
    return `${RNFS.DocumentDirectoryPath}/${MODEL_FILENAME}`;
}

/**
 * Check if the model is already downloaded.
 */
export async function isModelDownloaded() {
    const path = getModelPath();
    return RNFS.exists(path);
}

/**
 * Download the Whisper model with progress callback.
 * @param {function} onProgress - Callback with progress (0-1)
 * @returns {Promise<string>} - The file path of the downloaded model
 */
export async function downloadModel(onProgress) {
    const destPath = getModelPath();

    // Check if already downloaded
    const exists = await RNFS.exists(destPath);
    if (exists) {
        const stat = await RNFS.stat(destPath);
        // If file is larger than 10MB, assume it's valid
        if (stat.size > 10 * 1024 * 1024) {
            onProgress?.(1);
            return destPath;
        }
        // Otherwise delete and re-download
        await RNFS.unlink(destPath);
    }

    const downloadResult = RNFS.downloadFile({
        fromUrl: MODEL_URL,
        toFile: destPath,
        begin: () => { },
        progress: (res) => {
            const progress = res.bytesWritten / res.contentLength;
            onProgress?.(progress);
        },
        progressDivider: 5, // Report progress every 5%
    });

    const result = await downloadResult.promise;

    if (result.statusCode === 200) {
        onProgress?.(1);
        return destPath;
    } else {
        // Clean up partial download
        const partialExists = await RNFS.exists(destPath);
        if (partialExists) {
            await RNFS.unlink(destPath);
        }
        throw new Error(`Download failed with status: ${result.statusCode}`);
    }
}

/**
 * Delete the downloaded model.
 */
export async function deleteModel() {
    const path = getModelPath();
    const exists = await RNFS.exists(path);
    if (exists) {
        await RNFS.unlink(path);
    }
}

/**
 * Get model file size in MB (or null if not downloaded).
 */
export async function getModelSize() {
    const path = getModelPath();
    const exists = await RNFS.exists(path);
    if (!exists) return null;
    const stat = await RNFS.stat(path);
    return (stat.size / (1024 * 1024)).toFixed(1);
}
