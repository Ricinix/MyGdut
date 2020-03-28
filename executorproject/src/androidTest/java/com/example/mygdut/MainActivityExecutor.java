package com.example.mygdut;

import android.app.Activity;
import com.robotium.recorder.executor.Executor;

@SuppressWarnings("rawtypes")
public class MainActivityExecutor extends Executor {

	@SuppressWarnings("unchecked")
	public MainActivityExecutor() throws Exception {
		super((Class<? extends Activity>) Class.forName("com.example.mygdut.view.activity.MainActivity"),  "com.example.mygdut.R.id.", new android.R.id(), false, false, "1584891859849");
	}

	public void setUp() throws Exception { 
		super.setUp();
	}
}
