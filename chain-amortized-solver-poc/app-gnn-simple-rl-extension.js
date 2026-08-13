const C_LR=.003,C_EPOCHS=2,C_LOG=50;
let cp=null,cOpt=null,cPools=null,cLearned=null,cRandom=null;
const baseTrain=trainImitation,baseBench=runBenchmark;
function cParams(){return[...encParams(cp),cp.W1,cp.b1,cp.W2,cp.b2]}
function initC(){if(cOpt?.dispose)cOpt.dispose();cp={...enc('critic'),W1:makeVariable([2*EMBED_DIM+1,32],.08,'c1'),b1:makeBias(32,'c1b'),W2:makeVariable([32,1],.06,'c2'),b2:makeBias(1,'c2b')};cOpt=tf.train.adam(C_LR)}
function cloneS(s){return{n:s.n,assigned:Uint8Array.from(s.assigned),values:Int16Array.from(s.values),assignedCount:s.assignedCount}}
function cTensor(s,step){const h=embeddings(s,cp),z=tf.concat([h.mean(0),h.max(0),tf.tensor1d([step])]).reshape([1,2*EMBED_DIM+1]),q=tf.relu(tf.matMul(z,cp.W1).add(cp.b1));return tf.matMul(q,cp.W2).add(cp.b2).reshape([])}
function cPred(s,step){return tf.tidy(()=>cTensor(s,step).dataSync()[0])}
function trueCost(s){const d=Math.max(1,s.n-1);return violatedConstraintCount(s.values)/d+violationEnergy(s.values)/(DOMAIN_MAX*d)}
function learnedCand(s,r){const i=chooseRepairVariable(s,r,true,false);if(i==null)return null;return{i,v:learnedValue(s,i,r,true,false).v,src:'L'}}
function randomCand(s,r){return{i:Math.floor(r()*s.n),v:Math.floor(r()*(DOMAIN_MAX+1)),src:'R'}}
function applied(s,c,step){const q=cloneS(s);applyAction(q,c.i,c.v);return{...c,s:q,y:trueCost(q),step}}
function pool(s,r,nl,nr,step){const a=[];for(let k=0;k<nl;k++){const c=learnedCand(s,r);if(c)a.push(applied(s,c,step))}for(let k=0;k<nr;k++)a.push(applied(s,randomCand(s,r),step));if(!a.length){const c=learnedCand(s,r)||randomCand(s,r);a.push(applied(s,c,step))}return a}
function cLoss(a){let z=tf.scalar(0);for(const x of a)z=z.add(cTensor(x.s,x.step).sub(tf.scalar(x.y)).square());return z.div(a.length)}
async function cUpdate(a){let last=NaN;for(let e=0;e<C_EPOCHS;e++){const z=cOpt.minimize(()=>tf.tidy(()=>cLoss(a)),true,cParams());last=z.dataSync()[0];z.dispose()}await tf.nextFrame();return last}
