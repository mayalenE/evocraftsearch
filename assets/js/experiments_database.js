        var class_id_map = [];
        var show_from = 0;
        var exp_number = 5000;
        var show_images = 18;
        
        

        toggle_gif = function(name){
            if(name.includes('png')){ // toggle to gif
                return [name.replace("_end.png", ".gif"), true];
            }else{ // toggle to png
                return [name.replace(".gif", "_end.png"), false];
            }
        }
        
        $(document).ready(function(){

            class_id_map["All"]=[];
            for(i=0;i<exp_number;i++) class_id_map["All"].push(i);

            for(i = 0; i < show_images; i++){
                $("#show_result").append(
                    "<div class='evocraft_img' id='img_"+ (i-1) + "' style='position:relative;display:inline-block;text-align:center'>"+
                        "<span style='font-size:11px'></span><br>"+
                        "<img src='' style='width:200px'>"+
                        "<i class='fa fa-play-circle fa-3x' style='position:absolute;bottom:15px;right:10px;opacity:0.5'/>"+    
                    "</div>"
                );
            }
            $('#length_data').html(exp_number)
            $('#window_slider').attr('step',show_images)
        })
        $(document).ready(function(){

            $(".evocraft_img").click(function(){
                img_src = $(this).find('img').attr('src')
                img_src_spt = img_src.split('/');
                img_name = img_src_spt[img_src_spt.length - 1];
                [img_name, is_gif] = toggle_gif(img_name);
                img_src_spt[img_src_spt.length - 1] = img_name;
                img_src = img_src_spt.join('/')
                $(this).find('img').attr('src', img_src)
                if(is_gif==true){
                    $(this).find('i').attr('class', 'fa fa-stop-circle fa-3x')
                }else{
                    $(this).find('i').attr('class', 'fa fa-play-circle fa-3x')
                }
            })

            $.getJSON( "https://mayalenE.github.io/evocraftsearch/assets/experiments/experiment_names.json", function( data ) {
                var items = [];
                $.each( data, function( key, val ) {
                    $("#experiment").append( "<option value='" + val + "'>" + val + "</option>" );
                });
                displayImagesWithClassFromNumber(show_from, class_id_map["All"]);
            });


            $("#load_images").submit(function(){
                displayImagesWithClassFromNumber(show_from, class_id_map["All"]);
                /*
                $.getJSON("https://mayalenE.github.io/evocraftsearch/assets/experiments/"+ $("#experiment").val() +  "/repetition_" + pad( $("#repetition").val(),6) +"/classes.json", function( data ) {
                    $("#classes").html('<option value="All">All</option>');
                    $.each( data, function( key, val ) {
                        $("#classes").append( "<option value='" + key + "'>" + key + "</option>" );
                        class_id_map[key]= val;
                    });
                });*/
                $("#window_slider").val("0")
                $(".evocraft_img").find('i').attr('class', 'fa fa-play-circle fa-3x')
                return false;
            })

            $("#classes").change(function(){
                var index_window = $("#window_slider").val();
                displayImagesWithClassFromNumber(index_window,class_id_map[$("#classes").val()]);
            })  
            $("#window_slider").change(function(){
                var index_window = $("#window_slider").val();
                displayImagesWithClassFromNumber(index_window, class_id_map[$("#classes").val()]);
                $(".evocraft_img").find('i').attr('class', 'fa fa-play-circle fa-3x')
            })

            $("#step_back").click(function(){
                var index_window = parseInt($("#window_slider").val())-show_images;
                if(index_window < 0) index_window == 0;
                displayImagesWithClassFromNumber(index_window-show_images, class_id_map[$("#classes").val()])
                $("#window_slider").val(index_window)
                return false;
            })
            $("#step_forward").click(function(){
                var index_window = parseInt($("#window_slider").val())+show_images;
                if(index_window > exp_number) index_window == exp_number - show_images;
                displayImagesWithClassFromNumber(index_window, class_id_map[$("#classes").val()])
                $("#window_slider").val(index_window)
                return false;
            })

        })


        // start from
        function displayImagesWithClassFromNumber(start_from_image, image_map_for_class){
            // index of the image in the map
            index_in_map = image_map_for_class.indexOf(start_from_image);

            // if not found returns ind<0
            i = start_from_image;
            while(index_in_map < 0 && i >= 0){
                index_in_map = image_map_for_class.indexOf(i--);
            }

            // limit in between 0 ant number of experiments
            if(index_in_map > (exp_number-show_images)) index_in_map= exp_number-show_images;

            //show images
            for(i = 0; i < show_images; i++){
                var image_number = image_map_for_class[index_in_map + (i)];
                $("#img_"+(i-1)+" img").attr('src',"https://mayalenE.github.io/evocraftsearch/assets/experiments/"+ $("#experiment").val() +  "/repetition_" + pad( $("#repetition").val(),6) +"/images/run_" +  (image_number)  +"_rollout_end.png");
                $("#img_"+(i-1)+" span").html( image_number + 1);
            }
            $("#current_exp").html(index_in_map+"-"+(index_in_map+show_images))
        }


        function pad (str, max) {
            str = str.toString();
            return str.length < max ? pad("0" + str, max) : str;
        }
