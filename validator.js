<script type="text/javascript">
buttons = document.querySelectorAll("[name=colors]");
for (let button of Array.from(buttons)) {
button.addEventListener("change", () => {
document.body.style.background = button.value;
});
}
</script>

<script type="text/javascript">
$(document).ready(function(){
// reference the form element and watch for ’form’ submission event
$('form').submit(function(e){
// prevent the default browser behaviour on this case
e.preventDefault();
// reference 'this' (the form) then find the 'button'
// change its disabled attribute to 'disabled'
$(this).find('button').attr('disabled', 'disabled');
}); });
</script>

