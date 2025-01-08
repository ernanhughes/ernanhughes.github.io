+++
date = "2016-02-01T23:00:08Z"
title = "A simple android log class"

+++


This is a very simple log class I reuse in my projects

It is  a hybrid of [Timber](https://github.com/JakeWharton/timber "Timber") by Jake Wharton and the Log in 
[Android Universal Image Loader](https://github.com/nostra13/Android-Universal-Image-Loader "Android Universal Image Loader") by Sergey Tarasevich

<!--more-->

You can find the code here  [L](https://github.com/ernan/L "L")


```java

import android.util.Log;

public final class L {
    private static final String TAG = "L";
    private static final String FORMAT = "%1$s\n%2$s";
    private static volatile boolean debug = false;
    private static volatile boolean log = true;
    private L() {
    }

    public static void writeDebugLogs(boolean writeDebugLogs) {
        L.debug = writeDebugLogs;
    }

    public static void writeLogs(boolean writeLogs) {
        L.log = writeLogs;
    }

    public static void d(String message, Object... args) {
        if (debug) {
            log(Log.DEBUG, null, message, args);
        }
    }

    public static void i(String message, Object... args) {
        log(Log.INFO, null, message, args);
    }

    public static void w(String message, Object... args) {
        log(Log.WARN, null, message, args);
    }

    public static void e(Throwable ex) {
        log(Log.ERROR, ex, null);
    }

    public static void e(String message, Object... args) {
        log(Log.ERROR, null, message, args);
    }

    public static void e(Throwable ex, String message, Object... args) {
        log(Log.ERROR, ex, message, args);
    }

    private static void log(int priority, Throwable ex, String message, Object... args) {
        if (!log) return;
        if (args.length > 0) {
            message = String.format(message, args);
        }

        String log;
        if (ex == null) {
            log = message;
        } else {
            String logMessage = message == null ? ex.getMessage() : message;
            String logBody = Log.getStackTraceString(ex);
            log = String.format(FORMAT, logMessage, logBody);
        }
        Log.println(priority, TAG, log);
    }
}

```
