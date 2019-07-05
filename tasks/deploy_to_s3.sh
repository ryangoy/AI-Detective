cd web || exit 1
cd client || exit 1
npm install
npm run build
npm run deploy
