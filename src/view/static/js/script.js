$(document).ready(function () {
	var url = window.location;
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