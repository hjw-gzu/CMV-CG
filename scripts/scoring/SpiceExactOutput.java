package edu.anu.spice;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/** Java 8-compatible output adapter; all parsing and scoring remain in SPICE. */
public final class SpiceExactOutput {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Expected input JSON and new output JSON");
        }
        Path output = Paths.get(args[1]).toAbsolutePath();
        if (Files.exists(output)) {
            throw new IllegalArgumentException("Refusing to overwrite " + output);
        }
        SpiceArguments options = new SpiceArguments(new String[] {
            Paths.get(args[0]).toAbsolutePath().toString(), "-threads", "2", "-subset"
        });
        if (options.cache != null || options.outputPath != null || !options.synsets) {
            throw new IllegalStateException("Unexpected scoring options");
        }
        SpiceScorer scorer = new SpiceScorer();
        scorer.scoreBatch(options);
        Files.write(output, (scorer.stats.toJSONString() + "\n").getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
    }
}
