$(document).ready(function () {
	var url = window.location;
    $('ul.navbar-nav a').filter(function() {
         return this.href == url;
    }).parent().addClass('active').siblings().removeClass('active');
    /* Set the width of the side navigation to 250px */
    $("#user").mouseover(function() {
        $("#mySidenav").width(250);
        $("#main").css("marginLeft", 250+"px");
        $("#authorLeftNav").hide();
        $("#bookLeftNav").hide();
        $("#userLeftNav").show();
    });
    $("#author").mouseover(function() {
        $("#mySidenav").width(250);
        $("#main").css("marginLeft", 250+"px");
        $("#authorLeftNav").show();
        $("#bookLeftNav").hide();
        $("#userLeftNav").hide();
    });
    $("#book").mouseover(function() {
        $("#mySidenav").width(250);
        $("#main").css("marginLeft", 250+"px");
        $("#authorLeftNav").hide();
        $("#bookLeftNav").show();
        $("#userLeftNav").hide();
    });
    /*
    $('ul.navbar-nav a').mouseleave(function() {
        console.log("closeNav");
        $("#mySidenav").width(0);
    }); */
    /* Set the width of the side navigation to 0 */
    $('#closeLeftNav').click(function() {
        $("#mySidenav").width(0);
        $("#main").css("marginLeft", 0+"px");
        $("#authorLeftNav").hide();
        $("#bookLeftNav").hide();
        $("#userLeftNav").hide();
    });

    $("#prompt").onblur(function() {
        if($("#prompt").value.length > 0) { 
            $('#send-btn').disabled = false; 
        } else { 
            $('#send-btn').disabled = true;
        }
    })
    $('#prompt').onkeyup(function() {
        if($("#prompt").value.length > 0) { 
            $('#send-btn').disabled = false; 
        } else { 
            $('#send-btn').disabled = true;
        }
    });
    // Select your input type file and store it in a variable
    $("#chat-room-widget").addEventListener("send-btn", async (e) => {
        e.preventDefault();
        const prompt = $("#prompt").value;
        const image = $("#image");
        if (prompt.trim()) {
        var form = new FormData();
        form.append("prompt", prompt);
        if (image && image.files.length && image.files[0]) {
            //console.log(`Image name: ${image.files[0].name}, size: ${image.files[0].size}, type: ${image.files[0].type}`);
            form.append("image", image.files[0]);
        }// else 
            //console.log("No file selected!");
        // Display the key/value pairs
        /*for (var pair of form.entries()) {
            console.log(pair[0]+ ', ' + pair[1]); 
        }*/
        $('#submit').disabled = true;
        $("#submit").value = 'Processing...';
        const response = await fetch('/invoke', {
            method: 'POST',
            //headers: { 'Content-Type': 'multipart/form-data' }, Do NOT declare Content-Type: multipart/form-data in request header
            body: form
        });
        const data = await response.json();
        const queryContainer = document.createElement('div');
        queryContainer.innerHTML = `<div><strong>You:</strong> ${prompt}</div>`;
        $("#messages").appendChild(queryContainer);
        //$("#messages").innerHTML += `<div><strong>You:</strong> ${prompt}</div>`;
        var converter = new showdown.Converter();
        const responseContainer = document.createElement('div');
        responseContainer.innerHTML = `<strong>Gemini:</strong><div>${converter.makeHtml(data.response)}</div><br>`;
        $("#messages").appendChild(responseContainer);
        //$("#messages").innerHTML += `<div><strong>Gemini:</strong></div>${converter.makeHtml(data.message)}<br>`;
        $("#prompt").value = '';
        $("#image").value = '';
        $("#submit").value = 'Submit';
        } else 
        console.error(`Invalid prompt!`);
    });
    function createChatItem(message, sender) {
      var messages = document.getElementById("messages");
      if (sender === "") {
        content = `<p class="member-activity">${message}</p>`;
      } else {
        var senderIsUser = "{{user}}" === sender;
        var content = `
          <li class="message-item ${senderIsUser ? "self-message-item" : "peer-message-item"}">
              <p>${message}</p>
              <small class="${senderIsUser ? "muted-text" : "muted-text-white"}">${new Date().toLocaleString()}</small>
          </li>
      `;}
      messages.innerHTML += content;
    }
    function sendMessage() {
      var msgInput = document.getElementById("message-input");
      if (msgInput.value === "") return;
      var msg = msgInput.value;
      socketio.emit("message", { message: msg });
      msgInput.value = "";
    }    
});