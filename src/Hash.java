import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Random;

public class Hash
{
    public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
//        TODO: optimize

        MessageDigest md = MessageDigest.getInstance("SHA-1");
        String bits = stringToBits("hello world");
        System.out.println(bytesToHex(md.digest(new BigInteger(bits, 2).toByteArray())));
        System.out.println(sha1(bits)[80]);
//        System.out.println(checkSha1Impl());
//        encodeData(1, "./src/data.csv");
    }

    public static void encodeData(int size, String outFile) throws IOException {
        //512 bit strings
        //chance of overlap is so small that it doesn't matter
        //encoded in hex

        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        String[][] data = new String[size][81];
        //input, output, 79 other internal states from end-start
        for (int i = 0; i < size; i++) {
            data[i] = sha1(randBits());
            //https://www.baeldung.com/java-csv
        }

        pw.close();
    }

    public static String randBits() {
        Random rand = new Random();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 512; i++) {
            sb.append(rand.nextBoolean() ? '0' : '1');
        }
        return sb.toString();
    }

    public static String stringToBits(String str) {
        return new BigInteger(str.getBytes()).toString(2);
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

    public static boolean checkSha1Impl() throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-1");
        for (int i = 0; i < 100; i++) {
            String bits = randBits();
            if (!sha1(bits)[80].equals(bytesToHex(md.digest(new BigInteger(bits, 2).toByteArray())))) {
                System.out.println(i);
                System.out.println(sha1(bits)[80]);
                System.out.println(bytesToHex(md.digest(new BigInteger(bits, 2).toByteArray())));
                System.out.println(bits);
                return false;
            }
        }
        return true;
    }

    public static String[] sha1(String bits) {
        int h0 = 0x67452301;
        int h1 = 0xEFCDAB89;
        int h2 = 0x98BADCFE;
        int h3 = 0x10325476;
        int h4 = 0xC3D2E1F0;

        BigInteger message = new BigInteger(bits, 2);
        System.out.println(bits.length());
        System.out.println("hello world".getBytes().length*8);
        message = message.or(BigInteger.ONE.shiftLeft("hello world".getBytes().length*8));
        //1 in front to keep 0s

        BigInteger msgTemp = message;

        msgTemp = msgTemp.shiftLeft(1).or(BigInteger.ONE);

        while ((msgTemp.bitLength()-1) % 512 != 448) {
            msgTemp = msgTemp.shiftLeft(1);
        }
        msgTemp = msgTemp.shiftLeft(64).or(BigInteger.valueOf(message.bitLength()+1));

        String[] states = new String[81];
        states[0] = msgTemp.toString(16);

        for (int i = 0; i < msgTemp.bitLength()/512; i++) {
            int[] w = new int[80];
            for (int j = 0; j < 16; j++) {
                int shift = msgTemp.bitLength()-1 - (512*i + 32*j + 32);
                BigInteger mask = BigInteger.valueOf(Integer.MAX_VALUE)
                        .shiftLeft(1).or(BigInteger.ONE) //int max val is 32 bits
                        .shiftLeft(shift);
                w[j] = msgTemp.and(mask).shiftRight(shift).intValue();
            }
            for (int j = 16; j < 80; j++) {
                w[j] = leftRotate(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1);
            }

            int a = h0;
            int b = h1;
            int c = h2;
            int d = h3;
            int e = h4;

            for (int j = 0; j < 80; j++) {
                int f = 0; int k = 0;
                if (j >= 0 && j <= 19) {
                    f = (b & c) ^ (~b & d);
                    k = 0x5A827999;
                }
                else if (j >= 20 && j <= 39) {
                    f = b ^ c ^ d;
                    k = 0x6ED9EBA1;
                }
                else if (j >= 40 && j <= 59) {
                    f = (b & c) ^ (b & d) ^ (c & d);
                    k = 0x8F1BBCDC;
                }
                else if (j >= 60 && j <= 79) { //or just plain else
                    f = b ^ c ^ d;
                    k = 0xCA62C1D6;
                }

                //two's complement, signed/unsigned shouldn't matter for addition
                int temp = leftRotate(a, 5) + f + e + k + w[j];
                e = d;
                d = c;
                c = leftRotate(b, 30);
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
    private static int leftRotate(int num, int n) {
        //int is 32 bits
        return (num << n) | (num >>> (32 - n));
    }
    private static String attach(int[] parts) {
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
}
