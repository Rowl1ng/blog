/**
 * Created by luoling on 2017/4/9.
 */
var fish = new Array();
var Food = new Array(10);
var F;
var k = 0;
var evolving = true;

//var output = new PrintWriter();
var Time;

var Parent = new Array();
var Alive = new Array();
var n = 0;

var displaying = true;

function setup()
{
    background(0);
    createCanvas(windowWidth, windowHeight);
    Time = 0;
    //output = createWriter("evolution.txt");
    var initialP = loadStrings("brain.txt");
    var initialParent = new Array(initialP.length);
    for(var i = 0; i < initialP.length; i++){
        initialParent[i] = initialP[i];
    }
    var newWeight = initializeWeights(25,2,4,4,initialParent); //Brain(2,4),4
    fish = NewFish(newWeight);
    // newFish = new Creature[2];
    for(var i = 0; i < Food.length; i++){
        Food[i] = createVector(random(width),random(height));
    }
    frameRate(60);

    Parent = initializeWeights(25,2,4,4,initialParent);

    var AncestorWeight = new Array();

    for(var i = 0; i < 4; i++){
        AncestorWeight[i] = new Array();
        for(var j = 0; j < 4; j++){
            AncestorWeight[i][j] = new Array();
            for(var k = 0; k < 4; k++){
                AncestorWeight[i][j][k] = initialParent[i+j+k];
            }
        }
    }
    for(var i = 0; i < fish.length; i++){
        fish[i].brain.setWeight(AncestorWeight);
    }
}

function draw()
{
    if(displaying){
        background(255);
    }
    for(var i = 0; i < fish.length; i++){
        if(fish[i].isLiving()){
            fish[i].update();
            if(displaying){
                fish[i].display();
            }

            fish[i].Sense(Food, fish);
            fish[i].Think();

            Alive = fish[i].brain.getWeights();


            for(var j= 0; j < Food.length; j++){
                fill(255);
                if(displaying){
                    ellipse(Food[j].x, Food[j].y, 5,5);
                }
            }
            var k = fish[i].Eat(Food);
            if(k!=0){
                Food[k-1].set(random(width),random(height));
            }
        }
    }
    if(numLiving(fish) < 7 && evolving == true){
        for(var i = 0; i < fish.length; i++){
            if(!fish[i].dead){
                Parent[n%5] = fish[i].brain.getWeights();
                n++;
            }
        }
    }
    if(numLiving(fish) < 2 && evolving == true){
        background(0);
        //output.println(Time);
        fish = NewFish(crossOver(Parent, 25));
        Time = 0;
    }
    Time++;
}

function NewFish(weights)
{
    var newFish = new Array();
    for(var i = 0; i < weights.length; i++){
        newFish[i] = Creature.createNew(random(width), random(height));
        newFish[i].brain.setWeight(weights[i]);
    }
    return newFish;
}

function initializeWeights(a, b, c, d, initialParent)
{
    var newWeight = new Array();
    for(var i = 0; i < a; i++){
        newWeight[i] = new Array();
        for(var j = 0; j < b; j++){
            newWeight[i][j] = new Array();
            for(var k = 0; k < c; k++){
                newWeight[i][j][k] = new Array();
                for(var l = 0; l < d; l++){
                    if(i+j+k+l<initialParent.length){
                        newWeight[i][j][k][l] = initialParent[i+j+k+l];
                    }
                    else{
                        newWeight[i][j][k][l] = random(-1,1);
                    }
                }
            }
        }
    }
    return newWeight;
}

function numLiving(fish)
{
    var living = 0;
    for(var i = 0; i < fish.length; i++)
    {
        if(fish[i].isLiving()){
            living++;
        }
    }
    return living;
}

function crossOver(parent, n)
{
    var child = new Array();
    for(var i = 0; i < n; i++) {
        child[i] = new Array();
        for(var j = 0; j < parent[0].length; j ++) {
            child[i][j] = new Array();
            for(var k = 0; k < parent[0][0].length; k++) {
                child[i][j][k] = new Array();
                for(var l = 0; l < parent[0][0][0].length; l++)
                {
                    child[i][j][k][l] = parent[int(random(0, n))][j][k][l];
                }
            }
        }
    }
    for(var i = n-1; i < n+parent.length-1; i++)
    {
        child[i] = parent[i-(n-1)];
    }
    return child;
}


function keyPressed()
{
    if(keyCode == LEFT_ARROW)
    {
        for(var i = 0; i < Alive.length; i++){
            for(var j = 0; j < Alive[0].length; j++){
                for(var k = 0; k < Alive[0][0].length; k++){
                    print(Alive[i][j][k]);
                }
            }
        }
    }
    if(keyCode == DOWN_ARROW){
        displaying = !displaying;
        background(0);
    }

}

function mousePressed()
{
    Food[0] = createVector(random(width),random(height));
    //save("fish.jpg");
}

