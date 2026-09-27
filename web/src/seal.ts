// Opening what `export.seal` (src/xpfpl/export.py) encrypted: a key derived from the secret word
// (PBKDF2-SHA256) decrypts the data (AES-256-GCM) with the browser's WebCrypto. Nothing is sent
// anywhere; without the word the data can't be read.

export interface Sealed { v: number; iterations: number; salt: string; iv: string; data: string }

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

/** The sealed object, or null if the word is wrong (or the browser can't decrypt). */
export async function unseal<T>(sealed: Sealed, secret: string): Promise<T | null> {
  try {
    const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret.trim()), "PBKDF2", false, ["deriveKey"]);
    const key = await crypto.subtle.deriveKey(
      { name: "PBKDF2", salt: bytes(sealed.salt), iterations: sealed.iterations, hash: "SHA-256" },
      base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
    const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: bytes(sealed.iv) }, key, bytes(sealed.data));
    return JSON.parse(new TextDecoder().decode(plain)) as T;
  } catch {
    return null;
  }
}

const KEY = "xpfpl-my-team-word";

/** The word this browser last unlocked with (so the page opens unlocked next time), if any. */
export function rememberedWord(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function rememberWord(word: string | null): void {
  try {
    if (word === null) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, word);
  } catch {
    // private window or blocked storage: it just asks again next time
  }
}
