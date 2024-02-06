import prelim.Hash;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.security.NoSuchAlgorithmException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DataHandler {
    public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
        long start = System.nanoTime();
        encodeDataRestricted((int)Math.pow(10, 5), "./src/data-restricted-small.csv");
        long end = System.nanoTime();
        System.out.println((end-start)/Math.pow(10, 9) + " seconds to execute");
//        encodeData(100, "./src/test.csv");
    }

    public static void encodeData(int size, String outFile) throws IOException, NoSuchAlgorithmException {
        //chance of overlap is so small that it doesn't matter
        //encoded in hex

        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        //input, 80 internal states from start-end
        for (int i = 0; i < size; i++) {
            String bits = Hash.randBits(Hash.BIT_LENGTH);

            pw.println(convertToCSV(Hash.sha1(bits, 1)));
            //https://www.baeldung.com/java-csv
        }

        pw.close();
    }

    public static void encodeDataRestricted(int size, String outFile) throws IOException, NoSuchAlgorithmException {
        //chance of overlap is so small that it doesn't matter
        //encoded in hex

        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        //input, 80 internal states from start-end
        for (int i = 0; i < size; i++) {
            String bits = Hash.randTextBits(Hash.BIT_LENGTH);

            pw.println(convertToCSV(Hash.sha1(bits, 1)));
            //https://www.baeldung.com/java-csv
        }

        pw.close();
    }

    public static String convertToCSV(String[] data) {
        // special characters are not a worry since it's just hex data
        return Stream.of(data)
                .collect(Collectors.joining(","));
    }

}
