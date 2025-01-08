+++
date = "2016-03-28T23:45:48+01:00"
title = "Project 5: Dictator"

+++


Google have just released their speech API. One really cool feature is the ability to transcribe voice in real time.
Two years ago I built an app with this idea in mind. At that at the time I could no make it work. Now it is time to resurrect that app.
This post will cover the recording section of that application.

<!--more-->

This is a very long post so I will split it into sections

+[introduction](#Introduction)
+[icon](#Icon)
+[record](#Record Service)
+[main](#Main Fragment)
+[app](#Application)
+[play](#Player)
+[playfragment](#Player GUI)
+[calendar](#Calendar)
+[manage](#Manage Recordings)

## Introduction


This is a lot more complex that the previous application in may ways. Also there is a much larger amount of code.
This first post will introduce the app and will cover the GUI, in the next post I will integrate the Google speech back end
and later launch the final application. 

I have been made aware that these posts are kind of boring... I will do something about that .
For now lets get into the code. 

I will skim over the easy stuff and go into some details on parts of the application.



Essentially this app allows you to organize you voice notes or dictations. 
I will also convert these dictations into text using the new Google speech api.
The applications of this are literally endless....


This application uses some familiar code 

1. compile 'de.greenrobot:eventbus:2.4.0'
2. compile 'de.greenrobot:greendao:1.3.7'

And a few new ones.

3. compile 'com.squareup:android-times-square:1.6.5@aar'
4. compile 'io.reactivex:rxandroid:1.1.0'


Lets get started.


## Icon

First things first the Icon: 

I used this inkscape tutorial to build the icon.
[http://www.designmarkgraphics.co.uk/blog/articles/2013/11/26/create-long-shadow-icons-for-flat-designs-in-inkscape.html](Create long shadow icons for flat designs in inkscape)

This is a great Inkscape tutorial.


## Record Service


The application uses a service that listens for Record start and stop events. The service works with the main activity
using a broadcast receiver.

To do the  actual recording the app uses a [http://developer.android.com/reference/android/media/MediaRecorder.html] (MediaRecorder)
Once a recording has completed it stores the details about the recording in the data base.

```java

public class RecordService extends Service {

    public static final String BROADCAST_ACTION = "programmer.ie.dictator.recordservice";
    private final Handler handler = new Handler();
    Intent intent;
    MediaRecorder mRecorder;
    String mOutputFile;
    Recording recording;
    long mStartTime = 0l;

    private Runnable outGoingHandler = new Runnable() {
        public void run() {
            if (mRecorder != null) {
                updateRecordingInfo();
                handler.postDelayed(this, 100);
            }
        }
    };

    private MediaRecorder.OnErrorListener errorListener = new MediaRecorder.OnErrorListener() {
        @Override
        public void onError(MediaRecorder mr, int what, int extra) {
            L.e("Error: " + what + ", " + extra);
        }
    };

    private MediaRecorder.OnInfoListener infoListener = new MediaRecorder.OnInfoListener() {
        @Override
        public void onInfo(MediaRecorder mr, int what, int extra) {
            L.e("Warning: " + what + ", " + extra);
            if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED) {
                L.e("Maximum Duration Reached");
            } else if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_FILESIZE_REACHED) {
                L.e("Maximum File size Reached");
            }
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        EventBus.getDefault().register(this);
        intent = new Intent(BROADCAST_ACTION);
    }

    public void updateRecordingInfo() {
        Bundle b = new Bundle();
        long now = SystemClock.uptimeMillis();
        long totalTime = now - mStartTime;
        b.putLong(Util.DURATION, totalTime);
        int amp = mRecorder.getMaxAmplitude();
        b.putInt(Util.AMPLITUDE, amp);
        intent.putExtras(b);
        sendBroadcast(intent);
    }

    public void onEvent(RecordEvent event) {
        switch (event.action) {
            case Start: {
                mRecorder = new MediaRecorder();
                mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                mRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
                mRecorder.setOnErrorListener(errorListener);
                mRecorder.setOnInfoListener(infoListener);
                try {
                    String fileName = Util.getRecordingFileName(this);
                    File outputFile = new File(fileName);
                    mOutputFile = outputFile.getAbsolutePath();
                    mRecorder.setOutputFile(mOutputFile);
                    mRecorder.prepare();

                    recording = new Recording();
                    recording.setId(null);
                    recording.setName(Util.getRecordingName(this));
                    recording.setStartTime(new Date());
                    recording.setFileName(mOutputFile);

                    mStartTime = SystemClock.uptimeMillis();
                    handler.removeCallbacks(outGoingHandler);
                    handler.postDelayed(outGoingHandler, 100);
                    mRecorder.start();
                } catch (Exception ex) {
                    L.e(ex.getMessage());
                    return;
                }
                break;
            }
            case Stop: {
                if (mRecorder != null) {
                    try {
                        mRecorder.stop(); 
                    } catch (Exception ex) {
                        L.e(ex.getMessage());
                    }
                    mRecorder.reset();
                    mRecorder.release();
                    mRecorder = null;
                    recording.setEndTime(new Date());
                    File f = new File(mOutputFile);
                    L.i("File saved: " + mOutputFile + " " + f.length() + " (bytes)");
                    recording.setFileSize(f.length() / 1000);
                    Util.saveRecording(this, recording);
                    Util.addMediaEntry(this, mOutputFile);
                    Util.addCalendarEntry(this, recording);
                }
            }
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}

```


## Main Fragment<a name="main"></a>


The main fragment has a big record button and some other stuff to manage your recordings.
 
One nice touch is how it updates the notification area and your appbar with recording information.
I did not get the visualisation right yet, I will fix that shortly.

```java

public class MainFragment extends Fragment implements View.OnClickListener {
    TextView mRecordText;
    AudioEventView eventView;
    boolean isRecording = false;
    private BroadcastReceiver broadcastReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            Bundle b = intent.getExtras();
            int amplitude = b.getInt(Util.AMPLITUDE);
            eventView.addReading(amplitude);
            long totalTime = b.getLong(Util.DURATION);
            Spanned text = Html.fromHtml("<font color=\"#CF000F\">Recording " + DateTimeUtil.formatTime(totalTime / 1000) + "</font>");
            getActivity().getActionBar().setTitle(text);
            mRecordText.setText(text);
        }
    };

    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        final View rootView = inflater.inflate(R.layout.fragment_main, container, false);

        final ImageView mRecordButton = (ImageButton) rootView.findViewById(R.id.iconRecord);
        if (savedInstanceState != null && savedInstanceState.getSerializable(Util.RECORDING) != null) {
            isRecording = savedInstanceState.getBoolean(Util.RECORDING);
        }

        final ViewSwitcher viewSwitcher = (ViewSwitcher) rootView.findViewById(R.id.viewSwitcher);

        final Animation inAnimRight = AnimationUtils.loadAnimation(getActivity(),
                R.anim.grow_from_bottom);
        final Animation outAnimRight = AnimationUtils.loadAnimation(getActivity(),
                R.anim.fragment_slide_left_exit);
        viewSwitcher.setInAnimation(inAnimRight);
        viewSwitcher.setOutAnimation(outAnimRight);
        final View firstView = rootView.findViewById(R.id.mainLayout);
        final View secondView = rootView.findViewById(R.id.recordingFeedbackLayout);
        if (isRecording) {
            if (viewSwitcher.getCurrentView() != secondView) {
                viewSwitcher.showNext();
            }
        } else {
            if (viewSwitcher.getCurrentView() != firstView) {
                viewSwitcher.showPrevious();
            }
        }

        mRecordButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (viewSwitcher.getCurrentView() != secondView) {
                    viewSwitcher.showNext();
                }
                isRecording = true;
                EventBus.getDefault().post(new RecordEvent(RecordEvent.Action.Start));
            }
        });
        final ImageView mStopRecordButton = (ImageButton) rootView.findViewById(R.id.iconStopRecording);
        mStopRecordButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (viewSwitcher.getCurrentView() != firstView) {
                    viewSwitcher.showPrevious();
                }
                mRecordButton.setImageResource(R.drawable.record_start_half);
                getActivity().getActionBar().setTitle(R.string.app_name);
                EventBus.getDefault().post(new RecordEvent(RecordEvent.Action.Stop));
                isRecording = false;
            }
        });

        mRecordText = (TextView) rootView.findViewById(R.id.textRecord);
        eventView = (AudioEventView) rootView.findViewById(R.id.eventView);

        Intent ps = new Intent(getActivity(), RecordService.class);
        getActivity().startService(ps);
        return rootView;
    }

    @Override
    public void onResume() {
        super.onResume();
        getActivity().registerReceiver(broadcastReceiver, new IntentFilter(RecordService.BROADCAST_ACTION));
    }

    @Override
    public void onPause() {
        super.onPause();
        getActivity().unregisterReceiver(broadcastReceiver);
    }

    @Override
    public void onSaveInstanceState(Bundle savedInstanceState) {
        super.onSaveInstanceState(savedInstanceState);
        savedInstanceState.putBoolean(Util.RECORDING, isRecording);
    }

    public void onClick(View view) {
        ImageButton button = (ImageButton) view;
        String tag = (String) button.getTag();
        EventBus.getDefault().post(new SectionEvent(tag));
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.menu_item_share) {
            Intent share = new Intent(Intent.ACTION_SEND);
            share.setType("text/plain");
            startActivity(Intent.createChooser(share, getString(R.string.share_using)));
            return true;
        }
        if (id == R.id.menu_item_search) {
            Intent search = new Intent(getActivity(), SearchActivity.class);
            startActivity(search);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
}


```

## Search


There is a nice search feature which uses rx java which allows you to search for and play recordings. 
This uses an [http://developer.android.com/reference/android/widget/AutoCompleteTextView.html](AutoCompleteTextView) to show you an enhanced search set.


```java

public class SearchFragment extends Fragment {
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        final View rootView = inflater.inflate(R.layout.fragment_search, container, false);

        ImageButton b = (ImageButton) rootView.findViewById(R.id.searchButton);
        final AutoCompleteTextView tv = (AutoCompleteTextView) rootView.findViewById(R.id.searchText);
        final Func0 searchFunction = new Func0() {
            @Override
            public Object call() {
                final Hashtable<String, Object> searchResultItems = new Hashtable<>();
                String searchText = tv.getText().toString().toLowerCase();
                List<Recording> recordings = Util.getAllRecordings(getActivity());
                for (Recording r : recordings) {
                    String recordingName = r.getName().toLowerCase();
                    if (recordingName.contains(searchText)) {
                        searchResultItems.put(recordingName, r);
                    }
                }
                Binder<String> binder = new Binder.Builder<String>()
                        .addString(android.R.id.title, new StringExtractor<String>() {
                            @Override
                            public String getStringValue(String item, int position) {
                                return item;
                            }
                        })
                        .addString(android.R.id.content, new StringExtractor<String>() {
                            @Override
                            public String getStringValue(String item, int position) {
                                return item;
                            }
                        })
                        .addStaticImage(android.R.id.icon, new StaticImageLoader<String>() {
                            @Override
                            public void loadImage(String item, ImageView imageView, int position) {
                                Object result = searchResultItems.get(item);
                            }
                        }).build();
                List<String> searchResults = new ArrayList<>(searchResultItems.keySet());
                final SimpleAdapter<String> cardsAdapter = new SimpleAdapter<String>(getActivity(), searchResults, binder, R.layout.list_item_card);
                ListView cardsList = (ListView) rootView.findViewById(android.R.id.list);
                cardsList.setAdapter(cardsAdapter);
                cardsList.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                        String item = (String) searchResultItems.keySet().toArray()[position];
                        Object result = searchResultItems.get(item);
                        EventBus.getDefault().post(new PlayRecordingEvent((Recording) result));
                    }
                });
                return null;
            }
        };


        b.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                searchFunction.call();
            }
        });
        tv.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                searchFunction.call();
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        List<String> suggestionList = new ArrayList<>();
        tv.setThreshold(3);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(getActivity(), android.R.layout.simple_dropdown_item_1line, suggestionList);
        tv.setAdapter(adapter);
        adapter.setNotifyOnChange(true);
        return rootView;
    }
}

```


## Application


The application catches and logs exceptions. This I have found really useful.
So you build or run an app something happens and it dies.

This bit of code will write an entry to a database table showing what when where why etc.

Even better it only logs the last 20 exceptions so no fear of it draining resources on some ones phone
if you write a really buggy application.

It does this by adding an [http://developer.android.com/reference/java/lang/Thread.UncaughtExceptionHandler.html](UncaughtExceptionHandler) 
to itself. In the handler it writes the detaile to a database table.

This code depends on GreenDAO right now later I will remove this dependency and release the class as a single class gist. 

```java

public class DictatorApp extends Application implements Thread.UncaughtExceptionHandler {
    public static final String DATABASE_NAME = "dictator.db";

    public static void reportException(Context context, Throwable e) {
        ApplicationErrorReport report = new ApplicationErrorReport();
        report.packageName = report.processName = context.getPackageName();
        report.time = System.currentTimeMillis();
        report.type = ApplicationErrorReport.TYPE_CRASH;
        report.systemApp = false;

        ApplicationErrorReport.CrashInfo crash = new ApplicationErrorReport.CrashInfo();
        crash.exceptionClassName = e.getClass().getSimpleName();
        crash.exceptionMessage = e.getMessage();

        StringWriter writer = new StringWriter();
        PrintWriter printer = new PrintWriter(writer);
        e.printStackTrace(printer);

        crash.stackTrace = writer.toString();

        StackTraceElement stack = e.getStackTrace()[0];
        L.e(stack.toString());
        crash.throwClassName = stack.getClassName();
        L.e(stack.getClassName());
        crash.throwFileName = stack.getFileName();
        L.e(stack.getFileName());
        crash.throwLineNumber = stack.getLineNumber();
        L.e("Line Number: " + stack.getLineNumber());
        crash.throwMethodName = stack.getMethodName();
        L.e(stack.getMethodName());

        report.crashInfo = crash;

        ExceptionData data = new ExceptionData(null, report.packageName,
                report.time, Long.valueOf(report.type), crash.exceptionClassName, crash.exceptionMessage,
                crash.stackTrace, crash.throwClassName,
                crash.throwFileName,
                String.valueOf(crash.throwLineNumber),
                crash.throwMethodName);
        DaoMaster.DevOpenHelper helper = new DaoMaster.DevOpenHelper(context, "exceptions-db", null);
        DaoMaster daoMaster = new DaoMaster(helper.getWritableDatabase());
        DaoSession session = daoMaster.newSession();
        ExceptionDataDao dataDao = session.getExceptionDataDao();
        dataDao.insert(data);
        if (dataDao.count() > 20) {
            List<ExceptionData> items = dataDao.loadAll();
            dataDao.delete(items.get(0));
        }
        helper.close();
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Thread.setDefaultUncaughtExceptionHandler(this);
    }

    @Override
    public void uncaughtException(Thread thread, Throwable e) {
        reportException(this, e);
    }

}

```

## Player


There is also a media player here. There are many good posts on building a media player on the web so I wont go into to much details.

There service is  below.

I am sure you are starting to see a pattern, EventBus with a broadcast receiver.
The way I see it once you get it working rinse and repeat

```java

public class PlayService extends Service {

    public static final String BROADCAST_ACTION = "programmer.ie.dictator.playservice";
    static final int SKIP_TIME = 5000;
    final IBinder mBinder = new PlayBinder();
    private final Handler handler = new Handler();
    public MediaPlayer mPlayer;
    Intent intent;
    private Runnable outGoingHandler = new Runnable() {
        public void run() {
            updateRecordingInfo();
            handler.postDelayed(this, 200);
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        intent = new Intent(BROADCAST_ACTION);
        EventBus.getDefault().register(this);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        handler.removeCallbacks(outGoingHandler);
        handler.postDelayed(outGoingHandler, 100);
        return super.onStartCommand(intent, flags, startId);
    }

    public MediaPlayer getMediaPlayer() {
        return mPlayer;
    }

    public void updateRecordingInfo() {
        if (mPlayer != null && mPlayer.isPlaying()) {
            Bundle b = new Bundle();
            b.putInt(Util.DURATION, mPlayer.getDuration());
            b.putInt(Util.POSITION, mPlayer.getCurrentPosition());
            intent.putExtras(b);
            sendBroadcast(intent);
        }
    }

    public void onEvent(PlayEvent event) {
        switch (event.action) {
            case Start:
                L.d("Starting player " + mPlayer != null ? "!!! warning player not null" : "");
                String fileName = event.bundle.getString(Util.FILE_NAME);
                Uri uri = Uri.parse(fileName);
                mPlayer = MediaPlayer.create(this, uri);
                mPlayer.start();
                break;
            case Stop:
                L.d("Stopping player " + mPlayer == null ? "!!! warning player is null on stop." : "");
                if (mPlayer != null) {
                    mPlayer.reset();
                    mPlayer.release();
                    mPlayer = null;
                }
                break;
            case Pause:
                L.d("Pausing player ");
                if (mPlayer != null) {
                    mPlayer.pause();
                }
                break;
            case Resume:
                L.d("Resumeing player ");
                if (mPlayer != null) {
                    mPlayer.start();
                }
                break;
            case Seek:
                L.d("Seeking position ");
                int newPosition = event.bundle.getInt(Util.POSITION);
                if (mPlayer != null) {
                    mPlayer.seekTo(newPosition);
                }
                break;
            case Forward:
                L.d("Skipping ");
                if (mPlayer != null) {
                    int position = mPlayer.getCurrentPosition() + SKIP_TIME;
                    if (position > mPlayer.getDuration()) {
                        mPlayer.seekTo(0);
                    } else {
                        mPlayer.seekTo(position);
                    }
                }
                break;
            case Rewind:
                L.d("Rewinding ");
                if (mPlayer != null) {
                    int position = mPlayer.getCurrentPosition() - SKIP_TIME;
                    if (position < 0) {
                        mPlayer.seekTo(0);
                    } else {
                        mPlayer.seekTo(position);
                    }
                }
                break;
            case Restart:
                L.d("Rewinding ");
                if (mPlayer != null) {
                    mPlayer.seekTo(0);
                    mPlayer.start();
                }
                break;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    public class PlayBinder extends Binder {
        public PlayService getService() {
            return PlayService.this;
        }
    }
}

```

## Play Fragment

The play fragment is pretty simple also.
One thing i like here is the visualization: 

I grabbed this form this library [https://github.com/felixpalmer/android-visualizer](android-visualizer)

```java

public class PlayFragment extends Fragment {
    TextView mStartTime;
    TextView mEndTime;
    VisualizerView mVisualizer;
    SeekBar mSeekBar;
    private PlayService playerService;
    private BroadcastReceiver broadcastReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            Bundle b = intent.getExtras();
            int current = b.getInt(Util.POSITION);
            mStartTime.setText(DateTimeUtil.shortTimeFormat(current));
            int finalTime = b.getInt(Util.DURATION);
            mEndTime.setText(DateTimeUtil.shortTimeFormat(finalTime));
            mSeekBar.setProgress(current);
            mSeekBar.setMax(finalTime);
        }
    };
    private ServiceConnection mConnection = new ServiceConnection() {
        public void onServiceConnected(ComponentName className,
                                       IBinder binder) {
            PlayService.PlayBinder playerBinder = (PlayService.PlayBinder) binder;
            playerService = playerBinder.getService();
            mVisualizer.link(playerService.mPlayer);
            VisualiserFactory.addCircleBarRenderer(getActivity(), mVisualizer);
            VisualiserFactory.addBarGraphRenderers(getActivity(), mVisualizer);
        }

        public void onServiceDisconnected(ComponentName className) {
            playerService = null;
        }
    };

    @Override
    public void onResume() {
        super.onResume();
        Intent intent = new Intent(getActivity(), PlayService.class);
        getActivity().bindService(intent, mConnection,
                Context.BIND_AUTO_CREATE);
        getActivity().registerReceiver(broadcastReceiver, new IntentFilter(PlayService.BROADCAST_ACTION));
    }

    @Override
    public void onPause() {
        super.onPause();
        getActivity().unbindService(mConnection);
        getActivity().unregisterReceiver(broadcastReceiver);
    }

    @Override
    public void onDestroy() {
        EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Stop, new Bundle()));
        super.onDestroy();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        final View rootView = inflater.inflate(R.layout.fragment_play, container, false);
        mSeekBar = (SeekBar) rootView.findViewById(R.id.seekBar1);
        mEndTime = (TextView) rootView.findViewById(R.id.endTimeText);
        mStartTime = (TextView) rootView.findViewById(R.id.startTimeText);

        Intent ps = new Intent(getActivity(), PlayService.class);
        getActivity().startService(ps);

        Intent i = getActivity().getIntent();
        final String sUri = i.getExtras().getString(Util.FILE_NAME);
        EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Start, i.getExtras()));

        TextView songName = (TextView) rootView.findViewById(R.id.recordingTitle);
        songName.setText(Util.getShortName(sUri));

        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    Bundle b = new Bundle();
                    b.putInt(Util.POSITION, progress);
                    EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Seek, b));
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        final ImageView playButton = (ImageView) rootView.findViewById(R.id.playButton);
        playButton.setImageDrawable(getActivity().getResources().getDrawable(R.drawable.play_selected));
        final ImageView pauseButton = (ImageView) rootView.findViewById(R.id.pauseButton);
        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Resume, new Bundle()));
                playButton.setImageDrawable(getActivity().getResources().getDrawable(R.drawable.play_selected));
                pauseButton.setImageDrawable(getActivity().getResources().getDrawable(R.drawable.pause));
            }
        });
        pauseButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Pause, new Bundle()));
                playButton.setImageDrawable(getActivity().getResources().getDrawable(R.drawable.play));
                pauseButton.setImageDrawable(getActivity().getResources().getDrawable(R.drawable.pause_selected));
            }
        });
        final ImageView rewindButton = (ImageView) rootView.findViewById(R.id.rewindButton);
        rewindButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Rewind, new Bundle()));
            }
        });
        final ImageView forwardButton = (ImageView) rootView.findViewById(R.id.forwardButton);
        forwardButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Forward, new Bundle()));
            }
        });
        final ImageView restartButton = (ImageView) rootView.findViewById(R.id.restartButton);
        restartButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EventBus.getDefault().post(new PlayEvent(PlayEvent.Action.Restart, new Bundle()));
            }
        });
        mVisualizer = (VisualizerView) rootView.findViewById(R.id.visualizerView);

        return rootView;
    }
}

```


## Calendar

The calendar fragment is pretty cool it uses times square to organize your recordings.
It will show a highlighted calendar of you recordings allowing you to choose one of them frome each day.

```java

public class CalendarFragment extends Fragment {
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View rootView = inflater.inflate(R.layout.activity_calendar, container, false);
        final List<Recording> recordings = Util.getAllRecordings(getActivity());
        final CalendarPickerView calendar = (CalendarPickerView) rootView.findViewById(R.id.calendar_view);
        calendar.setOnDateSelectedListener(new CalendarPickerView.OnDateSelectedListener() {
            @Override
            public void onDateSelected(Date date) {
                L.d("Selecting date: " + date);
                calendar.selectDate(date);
            }

            @Override
            public void onDateUnselected(Date date) {
                final List<Recording> results = Util.getRecordingsForDate(getActivity(), date);
                CharSequence[] items = new CharSequence[results.size()];
                for (int i = 0; i < results.size(); ++i) {
                    Recording r = results.get(i);
                    items[i] = r.getName();
                }
                if (results.size() > 0) {
                    new AlertDialog.Builder(getActivity())
                            .setIcon(R.drawable.ic_save)
                            .setTitle("PlayEvent Recording for date " + DateTimeUtil.shortDateFormat(date))
                            .setItems(items, new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    if (results.size() > 0) {
                                        Recording recording = results.get(which);
                                        EventBus.getDefault().post(new PlayRecordingEvent((recording)));
                                    }
                                }
                            }).create().show();
                }
                List<Date> datesl = calendar.getSelectedDates();
                if (!datesl.contains(date)) {
                    calendar.selectDate(date);
                }
            }
        });

        final List<Date> dates = new ArrayList<>();
        if (recordings.size() > 0) {
            Recording first = recordings.get(recordings.size() - 1);
            Recording last = recordings.get(0);
            Calendar lastDate = DateTimeUtil.dateToCalendar(last.getStartTime());
            lastDate.add(Calendar.DAY_OF_WEEK, 7);
            int maxCount = 50;
            for (int i = 0; i < recordings.size() && i < maxCount; ++i) {
                Recording r = recordings.get(i);
                dates.add(r.getStartTime());
            }
            calendar.init(first.getStartTime(), lastDate.getTime()) //
                    .inMode(CalendarPickerView.SelectionMode.MULTIPLE)
                    .withSelectedDates(dates);
        } else {
            final Calendar nextMonth = Calendar.getInstance();
            nextMonth.add(Calendar.MONTH, 1);
            final Calendar lastMonth = Calendar.getInstance();
            lastMonth.add(Calendar.MONTH, -1);
            calendar.init(lastMonth.getTime(), nextMonth.getTime())
                    .inMode(CalendarPickerView.SelectionMode.SINGLE)
                    .withSelectedDate(new Date());
        }

        return rootView;
    }
}


```


## Recordings Management<a name="manage"a>

Last in this long post is the management fragment, This essentially allows you to manage your recordings.
You can share rename them 

```java


public class ManageFragment extends Fragment {
    SimpleAdapter<Recording> cardsAdapter;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View rootView = inflater.inflate(R.layout.fragment_manage, container, false);
        final List<Recording> items = Util.getAllRecordings(getActivity());
        final ListView listView = (ListView) rootView.findViewById(R.id.listview);
        final List<Recording> selected = new ArrayList<Recording>();
        Binder<Recording> binder = new Binder.Builder<Recording>()
                .addString(android.R.id.title, new StringExtractor<Recording>() {
                    @Override
                    public String getStringValue(Recording item, int position) {
                        return item.getName();
                    }
                })
                .addString(android.R.id.content, new StringExtractor<Recording>() {
                    @Override
                    public String getStringValue(Recording item, int position) {
                        return "Length: " + Util.getRecordingLength(item) + " Date: " +
                                DateTimeUtil.shortDateFormat(item.getStartTime());
                    }
                })
                .addCheckable(R.id.check, new BooleanExtractor<Recording>() {
                            @Override
                            public boolean getBooleanValue(Recording item, int position) {
                                return selected.contains(item);
                            }
                        }, new CheckedChangeListener<Recording>() {
                            @Override
                            public void onCheckedChangedListener(Recording recording, int position, View view, boolean b) {
                                if (selected.contains(recording)) {
                                    selected.remove(selected.indexOf(recording));
                                } else {
                                    selected.add(recording);
                                }
                            }
                        }
                )
                .addStaticImage(android.R.id.icon, new StaticImageLoader<Recording>() {
                    @Override
                    public void loadImage(Recording item, ImageView imageView, int position) {
                        Drawable d = Util.getImage(getActivity(), item);
                        imageView.setImageDrawable(d);
                    }
                }).build();

        cardsAdapter = new SimpleAdapter<Recording>(getActivity(), items, binder, R.layout.list_item_manage);
        listView.setAdapter(cardsAdapter);
        ImageButton editButton = (ImageButton) rootView.findViewById(R.id.editButton);
        editButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                for (final Recording recording : selected) {
                    AlertDialog.Builder alert = new AlertDialog.Builder(getActivity());
                    alert.setTitle("Rename recording");
                    alert.setMessage("Rename file: " + recording.getName());
                    final EditText input = new EditText(getActivity());
                    alert.setView(input);
                    alert.setPositiveButton("Rename",
                            new DialogInterface.OnClickListener() {
                                public void onClick(DialogInterface dialog,
                                                    int whichButton) {
                                    String value = input.getText().toString();
                                    Util.renameRecording(getActivity(), recording, value);
                                    cardsAdapter.notifyDataSetChanged();
                                }
                            }).create().show();
                }
            }
        });

        ImageButton deleteButton = (ImageButton) rootView.findViewById(R.id.deleteButton);
        deleteButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                for (final Recording recording : selected) {
                    Util.deleteRecording(getActivity(), recording);
                    items.remove(items.indexOf(recording));
                }
                selected.clear();
                cardsAdapter.notifyDataSetChanged();
            }
        });

        ImageButton sortButton = (ImageButton) rootView.findViewById(R.id.sortButton);
        sortButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Collections.reverse(items);
                cardsAdapter.notifyDataSetChanged();
            }
        });

        ImageButton detailsButton = (ImageButton) rootView.findViewById(R.id.detailsButton);
        detailsButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                for (final Recording recording : selected) {
                    // custom dialog
                    final Dialog dialog = new Dialog(getActivity());
                    dialog.setContentView(R.layout.dialog_details);
                    dialog.setTitle(recording.getName());

                    TextView text = (TextView) dialog.findViewById(R.id.startTime);
                    text.setText(DateTimeUtil.normalDateFormat(recording.getStartTime()));

                    text = (TextView) dialog.findViewById(R.id.endTime);
                    text.setText(DateTimeUtil.normalDateFormat(recording.getEndTime()));

                    text = (TextView) dialog.findViewById(R.id.fileSize);
                    text.setText(String.valueOf(recording.getFileSize()) + " KB");

                    text = (TextView) dialog.findViewById(R.id.filePath);
                    text.setText(recording.getFileName());
                    dialog.show();
                }
            }
        });

        ImageButton shareButton = (ImageButton) rootView.findViewById(R.id.shareButton);
        shareButton.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                for (final Recording recording : selected) {
                    Uri uri = Uri.parse(recording.getFileName());
                    Intent share = new Intent(Intent.ACTION_SEND);
                    share.setType("audio/*");
                    share.putExtra(Intent.EXTRA_STREAM, uri);
                    startActivity(Intent.createChooser(share, "Share Dictation"));
                }
            }
        });
        return rootView;
    }
}

```
