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
});