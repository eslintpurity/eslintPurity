
<script>
// begining of bootstrap accordion/collapsible
var acc = document.getElementsByClassName("accordion");
var i;

for (i = 0; i < acc.length; i++) {
acc[i].addEventListener("click", function() {
this.classList.toggle("active");
var panel = this.nextElementSibling;
if (panel.style.maxHeight){
panel.style.maxHeight = null;
} else {
panel.style.maxHeight = panel.scrollHeight + "px";
} 
});
}

// begining of linkin-to and activating bootstrap specific tab within the same page
$('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
    var target = this.href.split('#');
    $('.nav a').filter('a[href="#'+target[1]+'"]').tab('show');
})

// begining of linkin-to and activating bootstrap specific tab from external page
function getQuery(url) {
  var query = {},
      href = url || window.location.href;
  href.replace(/[?&](.+?)=([^&#]*)/g, function (_, key, value) {
      query[key] = decodeURI(value).replace(/\+/g, ' ');
  });
  return query;
}

function getParam(name){
    var obj = getQuery();
    return obj[name];
}

$(document).ready(function() {
  var target = getParam('target');
  $('.nav a[href="#'+target+'"]').tab('show');
});

// 
jQuery(document).ready(function($){
// Allow Page URL to activate a tab's ID
var taburl = document.location.toString();
if( taburl.match('#') ) {
$('.nav-tabs a[href="#'+taburl.split('#')[1]+'"]').tab('show');
}
// Allow internal links to activate a tab.
$('a[data-toggle="tab"]').click(function (e) {
e.preventDefault();
$('a[href="' + $(this).attr('href') + '"]').tab('show');
});
}); // End


var form = document.forms["contact"];
	form.addEventListener('submit',contact_submit,false);

	function contact_submit(e) {
		// Stop Form From Submitting
		e.preventDefault();

		// Set Initial Variables
		var target = e.target || e.srcElement;
		var to = 'doyinelugbadebo@gmail.com';
		var uri = 'mailto:' + to;
		var body = '';

		// Set Form Values to Variables
		var name = target.elements['name'].value;
		var email = target.elements['email'].value;
		var phone = target.elements['phone'].value;
		var reasons = target.elements['reasons'].value;
		var message = target.elements['message'].value;

		// Build Body / Message with all Input Fields
		body += message + "\r\n\r\n";
		body += "Name: " + name + "\r\n";
		body += "Phone Number: " + phone + "\r\n";
		body += "Reasons: " + reasons + "\r\n";

		// Build final Mailto URI
		uri += '?subject=' + encodeURIComponent(subject);
		uri += '&body=' + encodeURIComponent(body);

		// Open Mailto in New Window / Tab
		window.open(uri,'_blank');
	}
	
	$(document).ready(function(){
    $('[data-toggle="tooltip"]').tooltip(); 
});

function printFarmInventory(cows, chickens, pigs) {
console.log(`${zeroPad(cows, 3)} Cows`);
console.log(`${zeroPad(chickens, 3)} Chickens`);
console.log(`${zeroPad(pigs, 3)} Pigs`);
}
</script>
