package prelim;

import java.io.*;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Analyzer {
    public static void main(String[] args) throws IOException {
//        meanRoundHammingDistance("./src/test.csv", 100, "./src/hammingdist.out");
//        meanRoundHammingDistance("./src/data.csv", 100000, "./src/hammingdist.out");
//        compareDifferentials("./src/comparison.out");
        chiSquareOnInputs("./src/data.csv", "./src/inputrand.out");

//        System.out.println(Base64.getEncoder().encode(new BigInteger("010100100011111111100101101101010100110110000100000001010101111101110010010011011100100010001101111000000000010100100000111011000000000010000011101000000101001000010001001010110000010010101001111110010010011010000110110001111100110010011000010111001101001000000000111111111000001000110110001011101111111011000001111110001110111110010010011101100100101011110010111101011100001101101011000110100101001110011000011111101101011000000101"
//        , 2).toByteArray()));
    }

    public static void chiSquareOnInputs(String dataFile, String outFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        String line;

        // expected: 80
        double chiSquare = 0;
        while ((line = br.readLine()) != null) {
            StringTokenizer st = new StringTokenizer(line, ",");
            BigInteger input = new BigInteger(st.nextToken(), 16);
            int expected = input.bitLength()/2;
            chiSquare += (double)Math.pow(input.bitCount() - expected, 2)/expected;
            chiSquare += (double)Math.pow((input.bitLength()-input.bitCount()) - expected, 2)/expected;
        }
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        pw.println(chiSquare);
        pw.flush();
        pw.close();
    }

    public static void compareDifferentials(String outFile) throws IOException {
        String bits = Hash.randBits(Hash.BIT_LENGTH);
        String bits1 = (bits.charAt(0) == '0' ? "1" : "0") + bits.substring(1);
        String bits2 = bits1.charAt(0) + (bits.charAt(1) == '0' ? "1" : "0") + bits.substring(2);
        String bits3 = bits2.substring(0, 2) + (bits.charAt(2) == '0' ? "1" : "0") + bits.substring(3);
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        String[] rounds = Hash.sha1(bits, 1);
        String[] rounds1 = Hash.sha1(bits1, 1);
        String[] rounds2 = Hash.sha1(bits2, 1);
        String[] rounds3 = Hash.sha1(bits3, 1);
        pw.println(bits);
        for (int i = 1; i <= 80; i++) {
            pw.println(hammingDistance(rounds[i], rounds1[i]) + "\t" + hammingDistance(rounds[i], rounds2[i]) + "\t" +  hammingDistance(rounds[i], rounds3[i]));
        }
        pw.flush();
        pw.close();
    }

    public static void meanRoundHammingDistance(String dataFile, int length, String outFile) throws IOException {
        int h0 = 0x67452301;
        int h1 = 0xEFCDAB89;
        int h2 = 0x98BADCFE;
        int h3 = 0x10325476;
        int h4 = 0xC3D2E1F0;
        BigInteger og = new BigInteger(Hash.attach(new int[]{h0, h1, h2, h3, h4}), 16);

        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        double[] roundAvg = new double[80];
        String line;

        while ((line = br.readLine()) != null) {
            StringTokenizer st = new StringTokenizer(line, ",");
            st.nextToken();
            for (int i = 1; i <= 80; i++) {
                roundAvg[i-1] += og.xor(new BigInteger(st.nextToken(), 16)).bitCount();
                if (roundAvg[i-1] < 0) {
                    throw new RuntimeException("overflow!!");
                }
            }
        }
        roundAvg = Arrays.stream(roundAvg).map(d -> (double)d/length).toArray();
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        for (double d : roundAvg) {
            pw.println(d);
        }
        pw.flush();
        pw.close();
    }

    private static int hammingDistance(String h1, String h2) {
        BigInteger b1 = new BigInteger(h1, 16);
        BigInteger b2 = new BigInteger(h2, 16);
        return b1.xor(b2).bitCount();
    }
}
