
package org.viirya.graph;

import java.io.*;
import java.util.*;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.ArrayWritable;

public class DoubleArrayWritable extends ArrayWritable {
    public DoubleArrayWritable() {
        super(DoubleWritable.class);
    }
}

