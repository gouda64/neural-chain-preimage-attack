package prelim;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Random;

public class Hash
{
    public static final int BIT_LENGTH = 432;//448 - 8 - 8; // 2 8s to allow for padding, guaranteed only 1 block

    public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
//        System.out.println(checkSha1Impl());

        String bits = randTextBits(BIT_LENGTH);
        System.out.println(bits.length());
        int ones = 0;
        for (int i = 0; i < bits.length(); i++) {
            if (bits.charAt(i) == '1') ones++;
        }
        System.out.println(ones);
        System.out.println(bits);

//        int h0 = 0x67452301;
//        int h1 = 0xEFCDAB89;
//        int h2 = 0x98BADCFE;
//        int h3 = 0x10325476;
//        int h4 = 0xC3D2E1F0;
//        System.out.println(attach(new int[]{h0, h1, h2, h3, h4}));
    }

    public static boolean checkSha1Impl() throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-1");
        for (int i = 0; i < 1000; i++) {
            String bits = randBits(448-8-8);
            // randbits must be factor of 8
            String sha1 = sha1(bits, 1)[80];
            if (!sha1.equals(bytesToHex(md.digest(new BigInteger(bits, 2).toByteArray())))) {
                System.out.println(i);
                System.out.println(bits);
                System.out.println(new BigInteger(bits, 2).bitLength());
                System.out.println(sha1);
                System.out.println(bytesToHex(md.digest(new BigInteger(bits, 2).toByteArray())));
                return false;
            }
        }
        return true;
    }

    public static String[] sha1(String bits, int blocks) {
        // returns original message + internal states 1 to 80
        int h0 = 0x67452301;
        int h1 = 0xEFCDAB89;
        int h2 = 0x98BADCFE;
        int h3 = 0x10325476;
        int h4 = 0xC3D2E1F0;

        if (bits.charAt(0) == '1') {
            bits = "00000000" + bits;
        }

        BigInteger message = new BigInteger(bits, 2);
        message = message.or(BigInteger.ONE.shiftLeft(bits.length()));

        message = message.shiftLeft(1).or(BigInteger.ONE);
        message = message.shiftLeft((512 + (449 - message.bitLength()) % 512) % 512);
        message = message.shiftLeft(64).or(BigInteger.valueOf(bits.length()));

        if ((message.bitLength()-1)/512 < blocks) {
            throw new RuntimeException("too many blocks for message length");
        }
        if (blocks != -1 && (message.bitLength()-1)/512 > blocks) {
            System.out.println("less blocks than required");
        }

        String[] states = new String[81];
        states[0] = message.toString(16);

        for (int i = 0; i < (blocks == -1 ? (message.bitLength()-1)/512 : blocks); i++) {
            int[] w = new int[80];
            for (int j = 0; j < 16; j++) {
                int shift = message.bitLength()-1 - (512*i + 32*j + 32);
                BigInteger mask = BigInteger.valueOf(Integer.MAX_VALUE)
                        .shiftLeft(1).or(BigInteger.ONE)
                        .shiftLeft(shift);
                w[j] = message.and(mask).shiftRight(shift).intValue();
            }
            for (int j = 16; j < 80; j++) {
                w[j] = Integer.rotateLeft(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1);
            }

            int a = h0;
            int b = h1;
            int c = h2;
            int d = h3;
            int e = h4;

            for (int j = 0; j < 80; j++) {
                int f = 0; int k = 0;
                if (j >= 0 && j <= 19) {
                    f = (b & c) | (~b & d);
                    k = 0x5A827999;
                }
                else if (j >= 20 && j <= 39) {
                    f = b ^ c ^ d;
                    k = 0x6ED9EBA1;
                }
                else if (j >= 40 && j <= 59) {
                    f = (b & c) | (b & d) | (c & d);
                    k = 0x8F1BBCDC;
                }
                else if (j >= 60 && j <= 79) {
                    f = b ^ c ^ d;
                    k = 0xCA62C1D6;
                }

                int temp = Integer.rotateLeft(a, 5) + f + e + k + w[j];
                e = d;
                d = c;
                c = Integer.rotateLeft(b, 30);
                b = a;
                a = temp;

                states[j+1] = attach(new int[]{h0+a, h1+b, h2+c, h3+d, h4+e});
            }

            h0 += a;
            h1 += b;
            h2 += c;
            h3 += d;
            h4 += e;
        }

        return states;
    }

    // for testing
    public static String sha1Singular(String bits) {
        int h0 = 0x67452301;
        int h1 = 0xEFCDAB89;
        int h2 = 0x98BADCFE;
        int h3 = 0x10325476;
        int h4 = 0xC3D2E1F0;

        if (bits.charAt(0) == '1') {
            bits = "00000000" + bits; //padding
        }

        BigInteger message = new BigInteger(bits, 2);
        message = message.or(BigInteger.ONE.shiftLeft(bits.length()));

        message = message.shiftLeft(1).or(BigInteger.ONE);
        message = message.shiftLeft((512 + (449 - message.bitLength()) % 512) % 512);
        message = message.shiftLeft(64).or(BigInteger.valueOf(bits.length()));

        for (int i = 0; i < (message.bitLength()-1)/512; i++) {
            int[] w = new int[80];
            for (int j = 0; j < 16; j++) {
                int shift = message.bitLength()-1 - (512*i + 32*j + 32);
                BigInteger mask = BigInteger.valueOf(Integer.MAX_VALUE)
                        .shiftLeft(1).or(BigInteger.ONE)
                        .shiftLeft(shift);
                w[j] = message.and(mask).shiftRight(shift).intValue();
            }
            for (int j = 16; j < 80; j++) {
                w[j] = Integer.rotateLeft(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1);
            }

            int a = h0;
            int b = h1;
            int c = h2;
            int d = h3;
            int e = h4;

            for (int j = 0; j < 80; j++) {
                int f = 0; int k = 0;
                if (j >= 0 && j <= 19) {
                    f = (b & c) | (~b & d);
                    k = 0x5A827999;
                }
                else if (j >= 20 && j <= 39) {
                    f = b ^ c ^ d;
                    k = 0x6ED9EBA1;
                }
                else if (j >= 40 && j <= 59) {
                    f = (b & c) | (b & d) | (c & d);
                    k = 0x8F1BBCDC;
                }
                else if (j >= 60 && j <= 79) {
                    f = b ^ c ^ d;
                    k = 0xCA62C1D6;
                }

                int temp = Integer.rotateLeft(a, 5) + f + e + k + w[j];
                e = d;
                d = c;
                c = Integer.rotateLeft(b, 30);
                b = a;
                a = temp;
            }

            h0 += a;
            h1 += b;
            h2 += c;
            h3 += d;
            h4 += e;
        }

        return attach(new int[]{h0, h1, h2, h3, h4});
    }


    public static String attach(int[] parts) {
        String[] partStr = new String[parts.length];
        for (int i = 0; i < parts.length; i++) {
            partStr[i] = Integer.toBinaryString(parts[i]);
            while (partStr[i].length() != 32) {
                partStr[i] = "0" + partStr[i];
            }
        }

        StringBuilder sb = new StringBuilder();
        for (String s : partStr) {
            sb.append(s);
        }
        String bin = sb.toString();

        StringBuilder hex = new StringBuilder();
        for (int i = 0; i < bin.length(); i += 4) {
            hex.append(Integer.toHexString(Integer.parseInt(bin.substring(i, i+4), 2)));
        }
        return hex.toString();
    }

    public static String randTextBits(int bitLength) {
        String text = randText(bitLength/8);
        byte[] bytes = text.getBytes();

        StringBuilder binary = new StringBuilder();
        for (byte b : bytes)
        {
            int val = b;
            for (int i = 0; i < 8; i++)
            {
                binary.append((val & 128) == 0 ? 0 : 1);
                val <<= 1;
            }
        }
        return binary.toString();
    }
    private static String randText(int length) {
        final String source = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            sb.append(source.charAt((int)(Math.random() * source.length())));
        }
        return sb.toString();
    }

    public static String randBits(int length) {
        Random rand = new Random();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            sb.append(rand.nextBoolean() ? '0' : '1');
        }
        int traverse = 0;
        while (sb.substring(traverse, traverse + 8).equals("00000000")) {
            sb = new StringBuilder(sb.substring(traverse + 8));
            traverse += 8;
        }
        return sb.toString();
    }

    public static String stringToBits(String str) {
        String bits = new BigInteger(str.getBytes()).toString(2);
        while (bits.length() % 8 != 0) {
            bits = "0" + bits;
        }
        return bits;
    }

    private static final char[] HEX_ARRAY = "0123456789abcdef".toCharArray();
    public static String bytesToHex(byte[] bytes) {
        char[] hexChars = new char[bytes.length * 2];
        for (int j = 0; j < bytes.length; j++) {
            int v = bytes[j] & 0xFF;
            hexChars[j * 2] = HEX_ARRAY[v >>> 4];
            hexChars[j * 2 + 1] = HEX_ARRAY[v & 0x0F];
        }
        return new String(hexChars);
    }
}
