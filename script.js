/* Review Analysis */

function analyzeReview(){

let review=document.getElementById("review").value

fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({review:review})

})

.then(res=>res.json())

.then(data=>{

localStorage.setItem("sentiment",data.sentiment)

localStorage.setItem("confidence",data.confidence.toFixed(2))

window.location.href="/result"

})

}


/* Emoji Animation */

const emojis=["😀","😊","😍","😡","😢","🤖"]

const container=document.getElementById("emojiContainer")

function createEmoji(){

let e=document.createElement("div")

e.className="emoji"

e.innerText=emojis[Math.floor(Math.random()*emojis.length)]

e.style.left=Math.random()*100+"%"

e.style.animationDuration=(5+Math.random()*10)+"s"

container.appendChild(e)

setTimeout(()=>{e.remove()},15000)

}

setInterval(createEmoji,800)


/* Neural Network Background */

const canvas=document.getElementById("network")

const ctx=canvas.getContext("2d")

canvas.width=window.innerWidth
canvas.height=window.innerHeight

let nodes=[]

for(let i=0;i<80;i++){

nodes.push({

x:Math.random()*canvas.width,
y:Math.random()*canvas.height,
vx:(Math.random()-0.5),
vy:(Math.random()-0.5)

})

}

function animate(){

ctx.clearRect(0,0,canvas.width,canvas.height)

for(let i=0;i<nodes.length;i++){

let n=nodes[i]

n.x+=n.vx
n.y+=n.vy

if(n.x<0||n.x>canvas.width) n.vx*=-1
if(n.y<0||n.y>canvas.height) n.vy*=-1

ctx.beginPath()
ctx.arc(n.x,n.y,2,0,Math.PI*2)
ctx.fillStyle="white"
ctx.fill()

for(let j=i+1;j<nodes.length;j++){

let m=nodes[j]

let dist=Math.hypot(n.x-m.x,n.y-m.y)

if(dist<120){

ctx.beginPath()
ctx.moveTo(n.x,n.y)
ctx.lineTo(m.x,m.y)
ctx.strokeStyle="rgba(255,255,255,0.15)"
ctx.stroke()

}

}

}

requestAnimationFrame(animate)

}

animate()