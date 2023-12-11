import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.security.NoSuchAlgorithmException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DataHandler {
    public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
//        encodeData(100000, "./src/data.csv");
        encodeData(100, "./src/test.csv");
    }

    public static void encodeData(int size, String outFile) throws IOException, NoSuchAlgorithmException {
        //chance of overlap is so small that it doesn't matter
        //encoded in hex

        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        String[][] data = new String[size][81];
        //input, output, 79 other internal states from end-start
        for (int i = 0; i < size; i++) {
            String bits = Hash.randBits(Hash.BIT_LENGTH);
            data[i] = Hash.sha1(bits, 1);
            pw.println(convertToCSV(data[i]));
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
