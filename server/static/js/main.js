function toggle(id) {
	/* toggle element with id to be hidden or not. */
	var el = document.getElementById(id);
	if(el.style.display == 'none')
		el.style.display = 'block';
	else
		el.style.display = 'none';
}

function togglelink(id) {
	/* toggle element with id to be hidden or not, and also toggle
	 * link with id 'link'+id to start with 'show' or 'hide'. */
	var el = document.getElementById(id);
	var link = document.getElementById('link' + id);
	if(el.style.display == 'none') {
		el.style.display = 'block';
		link.innerHTML = 'hide' + link.innerHTML.substring(4);
	} else {
		el.style.display = 'none';
		link.innerHTML = 'show' + link.innerHTML.substring(4);
	}
}

function toggletextbox() {
	/* toggle a textbox to be single or multi line. */
	var state = document.queryform.textarea;
	var cur = document.queryform.query;
	var next = document.queryform.notquery;
	var link = document.getElementById('textboxlink');
	cur.name = 'notquery';
	cur.disabled = true;
	cur.style.display = 'none';
	next.name = 'query';
	next.disabled = false;
	next.style.display = 'block';
	if(state.disabled) {
		state.disabled = false;
		next.innerHTML = cur.value;
		link.innerHTML = 'small';
	} else {
		state.disabled = true;
		next.value = cur.value;
		link.innerHTML = 'large';
	}
}

function show(id, name) {
	/* show element with 'id' and enable (un-disable) all form elements with 'name'. */
	var el = document.getElementById(id);
	if(el.style.visibility != 'visible')
		el.style.visibility = 'visible';
	if(name != '') {
		var elems = document.getElementsByName(name);
		for (var n in elems)
			elems[n].disabled = false;
	}
}

function hide(id, name) {
	/* hide element with 'id' and disable all form elements with 'name'. */
	var el = document.getElementById(id);
	if(el.style.visibility != 'hidden')
		el.style.visibility = 'hidden';
	if(name != '') {
		var elems = document.getElementsByName(name);
		for (var n in elems)
			elems[n].disabled = true;
	}
}

function placeFocus() {
	/* place focus on first element of first form. */
	document.forms[0].elements[0].focus();
}

function triggerForm(name, val) {
	/* call the onChange event of the form element with 'name' and value 'val',
	 * so that the appropriate form elements may be shown/hidden. */
	var elems = document.getElementsByName(name)
	for (var n in elems)
		if(elems[n].value == val) {
			elems[n].onchange();
			break;
		}
}

function entsub(e) {
	/* call function if enter is pressed on form. */
	 var key;
	 if(window.event)
		key = window.event.keyCode;  // IE
	 else
		key = e.which;  // firefox
	 if(key == 13)
		ajaxFunction();
}

function ajaxFunction() {
	/* function to send request to parse a sentence and append the result to
	 * the current document. */
	var xmlhttp;
	if(window.XMLHttpRequest) {
		// code for IE7+, Firefox, Chrome, Opera, Safari
		xmlhttp=new XMLHttpRequest();
	} else if(window.ActiveXObject) {
		// code for IE6, IE5
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	} else {
		alert("Your browser does not support XMLHTTP!");
	}

	var div = document.createElement('div');
	div.innerHTML = '[...wait for it...]';
	document.getElementById('result').appendChild(div);
	// scroll to bottom of page
	window.scrollTo(0, document.body.scrollHeight);

	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			div.innerHTML = xmlhttp.responseText;
			var id = div.innerHTML.match(/id=([^ ]+) /);
			// collapse toggle-able items from here so that non-JS browsers
			// may view the contents
			if(id) {
				togglelink('n' + id[1]);
				togglelink('f' + id[1]);
				togglelink('i' + id[1]);
			}
			// scroll to bottom of page
			window.scrollTo(0, document.body.scrollHeight);
		}
	};
	var coarse = document.queryform.coarse;
	var objfun = document.queryform.objfun;
	var marg = document.queryform.marg;
	var est = document.queryform.est;
	var lang = document.queryform.lang;
	url = "parse?html=1&sent=" + encodeURIComponent(document.queryform.sent.value)
			+ "&coarse=" + encodeURIComponent(coarse.options[coarse.selectedIndex].value)
			+ "&objfun=" + encodeURIComponent(objfun.options[objfun.selectedIndex].value)
			+ "&marg=" + encodeURIComponent(marg.options[marg.selectedIndex].value)
			+ "&lang=" + encodeURIComponent(lang.options[lang.selectedIndex].value)
			+ "&est=" + encodeURIComponent(est.options[est.selectedIndex].value)
			;
	xmlhttp.open("GET", url, true);
	xmlhttp.send(null);
	document.queryform.sent.value = '';
}	

function checkall(name, val) {
	/* check / uncheck all checkboxes with 'name'. */
	var checkboxes = document.getElementsByName(name);
	for (var i in checkboxes)
		checkboxes[i].checked = val;
}

function numchecked() {
	/* update number of checked checkboxes and write to span
	 * 'numchecked'. */
	var checkboxes = document.getElementsByName('textcheck');
	var checked = 0;
	for (var i in checkboxes)
		if(checkboxes[i].name == 'textcheck' && checkboxes[i].checked)
			checked++;
	document.getElementById('numchecked').innerHTML = checked;
}

function mergecheckboxes() {
	/* given checkboxes identified by numbers, represent them succinctly
	 * as a series of intervals, e.g., '1-5,8-9,12' */
	var checkboxes = document.getElementsByName('textcheck');
	var n = 0;
	var start = 0;
	var result = ''
	while (n < checkboxes.length) {
		start = n;
		while (n < checkboxes.length && checkboxes[n].checked) {
			checkboxes[n].disabled = true;
			n++;
		}
		if(checkboxes[start].checked)
			if(start == n - 1)
				result += '.' + start;
			else
				result += '.' + start + '-' + n;
		while (n < checkboxes.length && !checkboxes[n].checked)
			n++;
	}
	document.forms[0].texts.value = result.substring(1);
	var radioboxes = document.getElementsByName('output');
	for (n in radioboxes) {
		if(radioboxes[n].checked) {
			document.forms[0].action = radioboxes[n].value;
			radioboxes[n].disabled = true;
			break;
		}
	}
	/* do not send empty query parameters to make URLs nicer. */
	for (name in ['limit', 'subset', 'target', 'target2']) {
		elem = document.getElementsByName(name);
		if (!elem.value) elem.disabled = true;
	}
}
