/**
 * Created by luoling on 2017/4/9.
 */
var Creature = {
    createNew: function(x, y)
    {
        var creature ={};
        creature.brain = Brain.createNew(2,4);
        creature.location = createVector(x, y);
        creature.velocity = createVector(random(-10, 10), random(-10, 10));
        creature.acceleration = createVector(0, 0);
        creature.R = 20.0;
        creature.r = 0.0
        creature.maxspeed = 4;
        creature.maxforce = 0.1;
        creature.oldHeading = creature.velocity.heading() + PI/2;
        creature.food = new Array();
        creature.food[0] = createVector(0, 0);
        creature.dead = false;
        //creature.newHeading = ;
        creature.Hunger = false;
        creature.sense = new Array();
        creature.fov = radians(30);
        creature.timeLiving = 0.0;
        creature.Color = color(random(255),random(255),random(255));
        creature.Contents = 30;
        creature.Capacity = 100;

        creature.checkEdges = function()
        {
            if (this.location.x > width)
            {
                this.location.x = 0;
            }
            else if (this.location.x < 0)
            {
                this.location.x = width;
            }

            if (this.location.y > height)
            {
                this.location.y = 0;
            }
            else if (this.location.y < 0)
            {
                location.y = height;
            }
        }

        creature.update = function()
        {
    //checkEdges();
            if (this.Contents < 0)
            {
                this.dead = true;
            }
            this.velocity.add(this.acceleration);
            this.velocity.limit(this.maxspeed);
            this.location.add(this.velocity);
            this.acceleration.mult(0);
            if (!this.dead)
            {
                this.timeLiving++;
            }
            this.Contents-=0.05;
        }

        creature.isLiving = function()
        {
            return !this.dead;
        }

        creature.getHunger = function()
        {
            return this.Hunger;
        }

        creature.ventral = function(h)
        {
            if (h > 0)
            {
                this.acceleration = createVector(cos(this.velocity.heading()), sin(this.velocity.heading()));
            } else if (h<0) {
                this.acceleration.setMag(this.acceleration.mag()-1);
            }
        }

        creature.caudal = function(a, b)
        {
            this.velocity = createVector(this.velocity.mag() * cos(this.velocity.heading()+a+b), this.velocity.mag() * sin(this.velocity.heading()+a+b));
        }

        creature.RightEye = function(food)
        {
            var visability = new Array();
            var distance = createVector(0,0);
            for (var i = 0; i < this.food.length; i++)
            {
                distance = p5.Vector.sub(this.food[i], this.location);
                visability[i] = abs((this.velocity.heading() - (this.fov/2) - distance.heading()) * distance.mag());
            }
            stroke(100);
    //line(location.x, location.y, location.x + 10*cos(velocity.heading()), location.y + 10*sin(velocity.heading()));
    //line(location.x, location.y, location.x + 10*cos(velocity.heading()- radians(15)), location.y + 10*sin(velocity.heading() - radians(15)));
            return min(visability);
        }

        creature.getLocation = function()
        {
            return this.location;
        }

        creature.Ear = function(fish)
        {
            var signal = 0;
            for (var i = 0; i < fish.length; i++)
            {
                var span = p5.Vector.sub(fish[i].getLocation(), this.location);
                if (span.mag() < 200)
                {
                    signal += span.mag();
                }
            }
            return signal;
        }

        creature.LeftEye = function(food)
        {
            var visability = new Array();
            var distance = createVector(0,0);
            for (var i = 0; i < food.length; i++)
            {
                distance = p5.Vector.sub(food[i], this.location);
                visability[i] = abs((this.velocity.heading() + (this.fov/2) - distance.heading() ) * distance.mag());
            }
            stroke(100);
    //line(location.x, location.y, location.x + 10*cos(velocity.heading() + radians(15)), location.y + 10*sin(velocity.heading() + radians(15)));
            return min(visability);
        }

        creature.Nose = function(food)
        {
            var distance = new Array();
            var scent = 0;
            for (var i = 0; i < food.length; i++)
            {
                distance[i] = pow((food[i].x - this.location.x), 2) + pow((food[i].y - this.location.y), 2);
                if (distance[i]<200)
                {
                    scent++;
                }
            }
            return scent;
        }

        creature.Stomach = function()
        {
            return 1/(1+exp(-1*(this.Capacity-this.Contents))) - 0.5;
        }

        creature.Sense = function(food, fish)
        {
            this.sense[0] = this.LeftEye(food);
            this.sense[1] = this.RightEye(food);
            this.sense[2] = this.Nose(food);
            this.sense[3] = this.Ear(fish);

        }

        creature.Think = function ()
        {
            var Action = this.brain.feedForward(this.sense);
            this.Act(Action);
        }

        creature.Act = function(action)
        {
            this.ventral(action[3]);
            this.caudal(action[0],action[1]);
        }

        creature.getContents = function()
        {
            return this.Contents;
        }

        creature.Eat = function(food)
        {
            var span = createVector(0,0);
            for (var i = 0; i < food.length; i++)
            {
                span = p5.Vector.sub(food[i], this.location);
                if (span.mag() < this.r)
                {
                    this.Contents+=5;
                    return i+1;
                }
            }
            return 0;
        }

        creature.display = function()
        {
            if (!this.dead)
            {
            var theta = this.velocity.heading() - PI/2;
            if (this.dead) {
                fill(255, 0, 0);
            } else {
                fill(this.Color);
            }
            this.r = this.R;
            stroke(0);
            push();
            translate(this.location.x, this.location.y);
            //text(Contents, 20, 20);
            rotate(theta);
            beginShape();
            vertex(0, this.r);
            vertex(this.r/2, 0);
            vertex(0, -this.r);
            vertex(this.r/3, -this.r-this.r/2);
            vertex(-this.r/3, -this.r-this.r/2);
            vertex(0, -this.r);
            vertex(-this.r/2, 0);
            endShape(CLOSE);
            pop();
            }
        }
        return creature;
    }
}