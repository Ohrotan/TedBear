//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("rec-btn");
var stopButton = document.getElementById("speak-btn");
var pauseButton = document.getElementById("pauseButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

//pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
    console.log("recordButton clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = {audio: true, video: false}

    /*
       Disable the record button until we get a success or fail from getUserMedia()
   */

    recordButton.disabled = true;
    stopButton.disabled = false;
    //pauseButton.disabled = false

    /*
        We're using the standard promise based getUserMedia()
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device
        */
        audioContext = new AudioContext({sampleRate: 16000});
        //update the format
        //document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /*
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, {numChannels: 1})

        //start the recording process
        rec.record()
        $("#speak-btn").removeClass("hide")
        $("#rec-btn").addClass("hide")
        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton.disabled = false;
        stopButton.disabled = true;
        //pauseButton.disabled = true
    });
}

function pauseRecording() {
    console.log("pauseButton clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton.innerHTML = "Pause";

    }
}

function stopRecording() {
    console.log("stopButton clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton.disabled = true;
    recordButton.disabled = false;
    //pauseButton.disabled = true;

    //reset button just in case the recording is stopped while paused
    //pauseButton.innerHTML="Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    $("#speak-btn").addClass("hide")
    $("#rec-btn").removeClass("hide")
    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink);

}

function createDownloadLink(blob) {


    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    li.style.listStyle = 'none'
    var link = document.createElement('a');


    var filename = new Date().toISOString();

    au.id = 'user_audio'
    au.controls = true;
    au.src = url;

    link.href = url;
    link.download = filename + ".wav";

    if (window.location.pathname == '/record') {
        li.appendChild(au)

        var fd = new FormData();
        fd.append("audio_data", blob, filename);

        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/record", true);
        xhr.send(fd);

        recordingsList.appendChild(li)
        return;
    }
    preAudio = document.getElementsByTagName('audio')
    /*
    if (preAudio.length > 0) {
        if(preAudio.item(0).id == "result_audio" &&preAudio.length == 2){
            preAudio.item(1).remove()
        }else{
            preAudio.item(0).remove()
        }
        for (i in recordingsList.children) {
            recordingsList.children.item(i).remove()
        }
        ted = document.getElementById("ted_words")
        user = document.getElementById("user_words")
        for (i in ted.children) {
            ted.children.item(i).remove()
            user.children.item(i).remove()
        }
    }
     */
//add the new audio element to li
    usercontrol = document.getElementById("user-control")
    usercontrol.appendChild(au);


//add the filename to the li
//li.appendChild(document.createTextNode(filename + ".wav "))

//add the save to disk link to li
//li.appendChild(link);

//upload link
    var upload = document.createElement('a');
    upload.href = "#";
    upload.innerHTML = "Upload";
    upload.addEventListener("click", function (event) {
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                //console.log( e.target.responseText);
                //console.log(e.target.responseText)

                console.log(e.target.responseText)
                result = e.target.responseText
                result = result.split("+++")

                tot = document.createElement('h3')
                tot.innerHTML = result[5]
                tot.style.color = 'var(--main-blue)'
                tot.style.textAlign = 'center'
                li.appendChild(tot)

                speed = document.createElement('h3')
                speed.innerHTML = result[0]
                speed.style.marginTop = '3rem'
                speed.style.textAlign = 'right'

                li.appendChild(speed)

                strength = document.createElement('h3')
                strength.innerHTML = result[1]
                strength.style.textAlign = 'right'
                strength.style.margin = '3rem 0rem -2rem 0rem'
                li.appendChild(strength)

                img_tag = document.createElement('img');
                img_tag.className = 'shadowing_result'
                //img_tag.src = '/static/graph/strength_result_' + e.target.responseText + '.png'
                img_tag.src = '/static/graph/strength_result_' + result[6] + '.png'
                li.appendChild(img_tag)

                pitch = document.createElement('h3')
                pitch.innerHTML = result[2]
                pitch.style.textAlign = 'right'
                pitch.style.margin = '3rem 0rem -2rem 0rem'
                li.appendChild(pitch)

                img_tag2 = document.createElement('img');
                img_tag2.className = 'shadowing_result'
                //img_tag2.src = '/static/graph/pitch_result_' + e.target.responseText + '.png'
                img_tag2.src = '/static/graph/pitch_result_' + result[6] + '.png'
                li.appendChild(img_tag2)

                word_point = document.getElementById('word_point')
                word_point.innerHTML = result[3]
                word_point.style.textAlign = 'right'

                li.appendChild(word_point)

                tedtable = document.getElementById("ted_words")
                usertable = document.getElementById("user_words")

                word = document.createElement('td')
                word.innerHTML = "TED: "
                tedtable.appendChild(word)

                word = document.createElement('td')
                word.innerHTML = "YOU: "
                usertable.appendChild(word)

                sentences = result[4].split('%%')
                ted = sentences[0].split('^^')
                user = sentences[1].split('^^')
                for (w in ted) {
                    word = document.createElement('td')
                    word.innerHTML = ted[w]
                    tedtable.appendChild(word)
                }
                for (w in user) {
                    word = document.createElement('td')
                    console.log(user[w], user[w].startsWith("@"))
                    if (user[w].startsWith("@")) {
                        word.style.color = 'red'
                        word.innerHTML = user[w].substring(1)
                    } else {
                        word.innerHTML = user[w]
                    }
                    usertable.appendChild(word)
                }


            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        fd.append("talks_id", document.getElementById("talks_id").value);
        fd.append("sentence_id", document.getElementById("sentence_id").value);
        fd.append("transcript_index", document.getElementById("transcript_index").value);
        xhr.open("POST", "/eval", true);
        xhr.send(fd);
    })
//li.appendChild(document.createTextNode(" "))//add a space in between
//li.appendChild(upload)//add the upload link to li
    upload.click()

//add the li element to the ol

    recordingsList.appendChild(li);
}