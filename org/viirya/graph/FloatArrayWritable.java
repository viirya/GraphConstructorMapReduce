
package org.viirya.graph;

import java.io.*;
import java.util.*;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.ArrayWritable;

public class FloatArrayWritable extends ArrayWritable {
    public FloatArrayWritable() {
        super(FloatWritable.class);
    }
}

