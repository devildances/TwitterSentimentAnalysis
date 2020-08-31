$(function(){
    $('#tweettext').keyup(function(){
        if($('#tweettext').val().length < 2){
            $('button').attr('disabled', true);
        }
        else {
            $('button').attr('disabled', false);
            $('.form-control-clear').fadeIn();
        }
        if($('#tweettext').val().length == 0) {
            $('.form-control-clear').hide();
        }
    });

    // Input type text clearable
    $('.has-clear input[type="text"]').on('input propertychange', function() {
        var $this = $(this);
        var visible = Boolean($this.val());
        $this.siblings('.form-control-clear').toggleClass('hidden', !visible);
    }).trigger('propertychange');

    $('.form-control-clear').click(function() {
        $(this).hide();
        $(this).siblings('input[type="text"]').val('')
        .trigger('propertychange').focus();
    });
});
